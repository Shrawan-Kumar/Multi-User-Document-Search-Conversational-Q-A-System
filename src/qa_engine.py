"""
qa_engine.py
------------
Wires together: FAISS retrieval (filtered by user access) + LLM (Gemini,
with automatic fallback to a local Ollama model on quota/rate-limit
errors) + per-user conversational memory, using LangChain's history-aware
retrieval pattern so follow-up questions ("What about their margins?")
get rewritten using prior chat context before retrieval runs.

Each user gets their own ConversationSession object held in a dict keyed
by email, so sessions never leak across users — this is what satisfies
the "queries from different users should not interfere" requirement.
"""

import os
import sys

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from access_control import build_faiss_filter, get_allowed_companies, AccessDeniedError


# ---------------------------------------------------------------------------
# Financial domain glossary
# ---------------------------------------------------------------------------
# Earnings releases use inconsistent terminology for the same metric (e.g.
# "PAT" vs "Net Profit" vs "Profit After Tax" vs "Group Net Profit"). Without
# this, a query for "PAT" can fail purely at the RETRIEVAL stage — the chunk
# containing the real number never gets pulled into context because the
# embedding similarity between "PAT" and "Group Net Profit" isn't strong
# enough on a small embedding model. We fix this upstream, in the query
# rewrite step, rather than only at answer-generation time.
FINANCIAL_GLOSSARY = (
    "PAT = Profit After Tax = Net Profit = Group Net Profit = Consolidated Net Profit. "
    "PBT = Profit Before Tax. "
    "EBITDA = Earnings Before Interest, Tax, Depreciation and Amortization = Operating Profit "
    "(sometimes reported as 'Segment Result' or 'EBITDA margin'). "
    "YoY = Year-on-Year = Year-over-Year. "
    "QoQ = Quarter-on-Quarter = Sequential. "
    "Revenue = Revenue from Operations = Total Income = Turnover = Net Sales. "
    "UVG = Underlying Volume Growth."
)


# ---------------------------------------------------------------------------
# Errors that should trigger a fallback to the secondary LLM provider.
# Google's SDK raises google.api_core.exceptions.ResourceExhausted for 429s;
# we also catch a few broader transient categories so a demo doesn't die on
# any single-provider hiccup (network blip, server overload, etc).
# ---------------------------------------------------------------------------
def _is_fallback_worthy(exc: Exception) -> bool:
    """True if this exception looks like a transient/quota issue worth
    retrying against the fallback provider, rather than a real bug."""
    try:
        from google.api_core.exceptions import (
            ResourceExhausted, ServiceUnavailable, DeadlineExceeded, InternalServerError
        )
        if isinstance(exc, (ResourceExhausted, ServiceUnavailable, DeadlineExceeded, InternalServerError)):
            return True
    except ImportError:
        pass

    # Fallback string-matching in case the specific exception class isn't
    # importable in this environment, or a different transport raises a
    # plain Exception with the relevant text in its message. Includes
    # connection-refused markers so "Ollama not running" (now the default
    # primary) also triggers fallback to Gemini, not just Gemini-side errors.
    message = str(exc).lower()
    transient_markers = [
        "resourceexhausted", "429", "quota", "rate limit",
        "rate_limit", "503", "unavailable", "deadline exceeded",
        "connecterror", "connection refused", "10061", "refused",
        "max retries exceeded", "failed to establish a new connection",
    ]
    return any(marker in message for marker in transient_markers)


def _short_error_label(exc: Exception) -> str:
    """
    Translate an exception into a short, demo-friendly label for the UI.
    The full exception (with quota IDs, retry-delay payloads, etc.) is
    still printed to the terminal for debugging — this is just what gets
    shown on screen.
    """
    message = str(exc).lower()
    if "quota" in message or "resourceexhausted" in message or "429" in message:
        return "rate limit / quota reached"
    if "503" in message or "unavailable" in message:
        return "service temporarily unavailable"
    if "deadline" in message or "timeout" in message:
        return "request timed out"
    if "connecterror" in message or "10061" in message or "refused" in message:
        return "could not connect to the model server"
    return exc.__class__.__name__ or "transient error"


# ---------------------------------------------------------------------------
# Shared resources (loaded once, reused across all user sessions)
# ---------------------------------------------------------------------------
_embeddings = None
_vectorstore = None
_primary_llm = None
_fallback_llm = None  # False (sentinel) once we've tried and it's unavailable


def _build_groq_llm():
    from langchain_groq import ChatGroq

    if not config.GROQ_API_KEY:
        raise EnvironmentError(
            "LLM_PROVIDER is 'groq' but no GROQ_API_KEY was found. "
            "Get a free key at https://console.groq.com and add it to .env:\n"
            "  GROQ_API_KEY=your-key-here"
        )
    return ChatGroq(
        model=config.GROQ_MODEL_NAME,
        api_key=config.GROQ_API_KEY,
        temperature=0.1,
    )


def _build_google_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not config.GOOGLE_API_KEY:
        raise EnvironmentError(
            "LLM_PROVIDER is 'google' but no GOOGLE_API_KEY was found. "
            "Get a free key at https://aistudio.google.com/apikey and add it to .env:\n"
            "  GOOGLE_API_KEY=your-key-here"
        )
    os.environ["GOOGLE_API_KEY"] = config.GOOGLE_API_KEY
    return ChatGoogleGenerativeAI(
        model=config.GOOGLE_MODEL_NAME,
        google_api_key=config.GOOGLE_API_KEY,
        transport="rest",
        temperature=0.1,
    )


def _build_ollama_llm():
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=config.OLLAMA_MODEL_NAME,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.1,
    )


def _build_llm_by_provider(provider: str):
    if provider == "groq":
        return _build_groq_llm()
    elif provider == "google":
        return _build_google_llm()
    elif provider == "ollama":
        return _build_ollama_llm()
    else:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Expected 'groq', 'google', or 'ollama'."
        )


def _fallback_provider() -> str:
    """Return the configured fallback provider name."""
    return config.FALLBACK_PROVIDER


def _load_shared_resources():
    """
    Loads embeddings, the FAISS index, and the LLM(s).

    When config.ENABLE_LLM_FALLBACK is True, this also eagerly builds the
    secondary (fallback) LLM client up front — so the FIRST time a 429
    happens mid-demo, there's no extra delay/risk from lazily constructing
    a fresh client under pressure. If the fallback provider fails to build
    (e.g. Ollama not running), we log a warning but don't crash startup —
    fallback simply won't be available until it is.
    """
    global _embeddings, _vectorstore, _primary_llm, _fallback_llm

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    if _vectorstore is None:
        if not os.path.exists(config.FAISS_INDEX_PATH):
            raise FileNotFoundError(
                f"No FAISS index found at {config.FAISS_INDEX_PATH}. "
                f"Run `python src/ingest.py` first."
            )
        _vectorstore = FAISS.load_local(
            config.FAISS_INDEX_PATH,
            _embeddings,
            allow_dangerous_deserialization=True,
        )

    if _primary_llm is None:
        _primary_llm = _build_llm_by_provider(config.LLM_PROVIDER)

    if config.ENABLE_LLM_FALLBACK and _fallback_llm is None:
        fallback_prov = _fallback_provider()
        try:
            _fallback_llm = _build_llm_by_provider(fallback_prov)
            print(f"[qa_engine] Fallback provider ready: '{fallback_prov}'")
        except Exception as e:
            print(f"[qa_engine] WARNING: could not initialize fallback provider "
                  f"'{fallback_prov}' ({e}). Fallback will be unavailable "
                  f"until this is fixed.")
            _fallback_llm = False

    return _embeddings, _vectorstore, _primary_llm, _fallback_llm


# ---------------------------------------------------------------------------
# Per-user conversational session
# ---------------------------------------------------------------------------
class UserSession:
    """
    One isolated conversational session per logged-in user.

    - `chat_history` holds only this user's turns (in-memory list of
      LangChain message objects). Nothing here is shared globally.
    - `retriever` is built with an access-control filter baked in, so
      this user's queries can structurally never return another
      company's chunks, regardless of what they ask.
    - Two RAG chains are built (primary + fallback, if enabled), sharing
      the same retriever/prompts but backed by different LLMs. `.ask()`
      tries the primary chain first and transparently retries against the
      fallback chain if the primary call fails with a quota/transient error.
    """

    def __init__(self, user_email: str):
        self.user_email = user_email.strip().lower()
        self.allowed_companies = get_allowed_companies(self.user_email)
        self.chat_history: list = []

        _, vectorstore, primary_llm, fallback_llm = _load_shared_resources()
        self.primary_provider = config.LLM_PROVIDER
        self.fallback_provider = _fallback_provider()
        self.fallback_available = bool(fallback_llm)

        filter_fn, _ = build_faiss_filter(self.user_email)
        base_retriever = vectorstore.as_retriever(
            search_kwargs={"k": config.TOP_K, "filter": filter_fn}
        )

        self.rag_chain = self._build_rag_chain(primary_llm, base_retriever)
        if self.fallback_available:
            self.fallback_rag_chain = self._build_rag_chain(fallback_llm, base_retriever)
        else:
            self.fallback_rag_chain = None

    def _build_rag_chain(self, llm, base_retriever):
        """Build a full history-aware RAG chain for a given LLM instance."""
        # --- History-aware retriever: rewrites follow-up questions using
        # chat history into a standalone query before hitting the vector store.
        # The financial glossary is included HERE (not just in the answer
        # prompt) because retrieval quality depends on the query text itself
        # — if "PAT" never gets expanded toward "Net Profit" before the
        # similarity search runs, the correct chunk may never be retrieved
        # at all, regardless of how good the downstream LLM is.
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given a chat history and the latest user question which might "
             "reference context in the chat history, formulate a standalone "
             "question which can be understood without the chat history. "
             "If the question uses a financial acronym or abbreviation, "
             "expand it using this glossary so retrieval can find the right "
             f"terminology: {FINANCIAL_GLOSSARY} "
             "Do NOT answer the question, just reformulate it if needed, "
             "otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            llm, base_retriever, contextualize_prompt
        )

        # --- Answer-generation prompt, scoped explicitly to this user's companies.
        # NOTE: {context} must remain a plain string literal (not inside an
        # f-string) so LangChain registers it as a template input variable.
        allowed_str = ', '.join(self.allowed_companies)
        system_message = (
            f"You are a strict financial document assistant. "
            f"The current user ({self.user_email}) is authorized to access "
            f"documents ONLY for these companies: {allowed_str}. "
            f"Financial terminology glossary: {FINANCIAL_GLOSSARY} "
            "STRICT RULES you must follow without exception:\n"
            "1. Every number or fact you state MUST come from a retrieved chunk "
            "that explicitly names the company it belongs to. "
            "NEVER attribute a figure to a company unless that chunk clearly "
            "identifies that company as the source.\n"
            "2. If the retrieved chunks contain data from multiple companies, "
            "answer ONLY about the company the user asked about. "
            "Do not blend or cross-attribute figures across companies.\n"
            "3. If the user asks about a company whose data is NOT present "
            "in the retrieved context, respond with exactly ONE sentence: "
            f"'<CompanyName> data is not available in your authorized documents "
            f"(you have access to: {allowed_str}).'\n"
            "4. Never speculate, infer, or use outside knowledge. "
            "If the context does not explicitly state it, say you do not know.\n"
            "5. Do not explain how the system works internally.\n"
            "\nContext:\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        document_chain = create_stuff_documents_chain(llm, qa_prompt)

        return create_retrieval_chain(history_aware_retriever, document_chain)

    def ask(self, question: str) -> dict:
        """
        Submit a query, get back an answer + source chunks + which provider
        actually served the response. Tries the primary provider first;
        on a quota/rate-limit/transient error, transparently retries the
        SAME question against the fallback provider (if available).
        """
        used_provider = self.primary_provider
        fell_back = False
        fallback_reason = None

        try:
            result = self.rag_chain.invoke({
                "input": question,
                "chat_history": self.chat_history,
            })
        except Exception as e:
            if self.fallback_available and _is_fallback_worthy(e):
                # Full detail goes to the terminal only — useful for you,
                # not something an interviewer watching the screen needs to see.
                print(f"[qa_engine] '{self.primary_provider}' call failed "
                      f"({e.__class__.__name__}: {e}). Falling back to "
                      f"'{self.fallback_provider}' for this query...")
                used_provider = self.fallback_provider
                fell_back = True
                fallback_reason = _short_error_label(e)  # short label only, for the UI
                result = self.fallback_rag_chain.invoke({
                    "input": question,
                    "chat_history": self.chat_history,
                })
            else:
                # Not a transient error, or no fallback available — surface it.
                raise

        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=result["answer"]))

        sources = [
            {
                "company": doc.metadata.get("company"),
                "source_file": doc.metadata.get("source_file"),
                "page": doc.metadata.get("page", "N/A"),
                "snippet": doc.page_content[:300],
            }
            for doc in result.get("context", [])
        ]

        return {
            "answer": result["answer"],
            "sources": sources,
            "provider_used": used_provider,
            "fell_back": fell_back,
            "fallback_reason": fallback_reason,
        }

    def reset_memory(self):
        self.chat_history = []


# ---------------------------------------------------------------------------
# Session registry — keeps each user's session fully isolated
# ---------------------------------------------------------------------------
_active_sessions: dict[str, UserSession] = {}


def get_or_create_session(user_email: str) -> UserSession:
    user_email = user_email.strip().lower()
    if user_email not in _active_sessions:
        _active_sessions[user_email] = UserSession(user_email)
    return _active_sessions[user_email]


def clear_session(user_email: str):
    user_email = user_email.strip().lower()
    if user_email in _active_sessions:
        del _active_sessions[user_email]