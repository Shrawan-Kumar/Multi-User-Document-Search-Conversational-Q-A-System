# Multi-User Document Search & Conversational Q&A System

A multi-tenant Retrieval-Augmented Generation (RAG) system that lets multiple
users query a shared set of earnings-call PDFs, while enforcing **per-user,
per-company access control** and maintaining **isolated conversational memory**
per user.

Built with:
**LangChain · FAISS · HuggingFace sentence-transformers (local embeddings) ·
Groq API / Llama 3.3 70B (primary LLM) · Ollama / Phi-3 (local fallback LLM) ·
Streamlit (UI)**

---

## 1. Architecture

```
                 ┌──────────────────────┐
                 │   data/pdfs/*.pdf     │  5 Tata company earnings docs
                 └──────────┬───────────┘
                            │  ingest.py
                            ▼
          ┌──────────────────────────────────────┐
          │ Load  →  tag each chunk               │
          │          metadata["company"] = "TCPL" │
          │ Chunk (RecursiveCharacterTextSplitter) │
          │ Embed (all-MiniLM-L6-v2, local,        │
          │        no API key needed)              │
          └──────────────┬───────────────────────┘
                         ▼
                ┌──────────────────┐
                │  FAISS index      │  vectorstore/faiss_index/
                │  (on disk)        │  index.faiss + index.pkl
                └────────┬──────────┘
                         │
   ┌─────────────────────┴──────────────────────┐
   │               qa_engine.py                   │
   │                                               │
   │  UserSession(email)                           │
   │   ├─ allowed_companies = ACCESS_MAP[email]    │
   │   ├─ retriever = FAISS.as_retriever(          │  <-- access control
   │   │      filter = lambda meta: meta.company   │      enforced HERE,
   │   │               in allowed_companies)       │      before retrieval
   │   ├─ history_aware_retriever                  │      unauthorized chunks
   │   │    (rewrites follow-ups using             │      never reach LLM
   │   │     this user's chat_history)             │
   │   ├─ primary_rag_chain  → Groq (cloud, fast)  │
   │   ├─ fallback_rag_chain → Ollama (local)      │
   │   └─ chat_history: list (per-user, isolated)  │
   └─────────────────────┬──────────────────────┘
                         ▼
                ┌──────────────────┐
                │   app.py (UI)     │  Streamlit: login · chat · sources
                └──────────────────┘
```

**Why isolation is structurally guaranteed, not just UI-hidden:**
The FAISS `filter` runs *inside* the similarity search
(`access_control.build_faiss_filter`). An unauthorized chunk is never
retrieved, never placed into the LLM context window, and can never appear
in an answer — regardless of how the question is phrased or which LLM
is serving the response.

---

## 2. LLM Provider Setup

The system uses a **pluggable, three-provider architecture**:

| Role | Provider | Model | Cost |
|---|---|---|---|
| **Primary** | Groq API | llama-3.3-70b-versatile | Free tier |
| **Fallback** | Ollama (local) | phi3 | Free, offline |
| **Optional** | Google AI Studio | gemini-2.0-flash | Requires funded balance |

Switch providers by changing `LLM_PROVIDER` and `FALLBACK_PROVIDER` in
`src/config.py` — no other code changes needed.

### Automatic Fallback

`ENABLE_LLM_FALLBACK = True` (default): if the primary provider hits a
quota/rate-limit/connection error, the system transparently retries the
**same query** against the fallback provider and shows a calm info banner
in the UI. The full error detail is logged to the terminal only — the
interviewer/user sees a clean one-line notice, not a stack trace.

---

## 3. Setup

### Prerequisites
- Python 3.10+
- **Groq API key** (free, no billing required) — sign up at
  [console.groq.com](https://console.groq.com)
- **Ollama** installed locally (for fallback) — [ollama.com](https://ollama.com)

### Steps

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd doc-qa-system

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Open .env and fill in your Groq key:
#   GROQ_API_KEY=gsk_your_key_here
# .env is git-ignored and will never be committed to GitHub.

# 5. Set up Ollama fallback (one-time)
ollama pull phi3                  # ~2.2 GB download
# Ollama auto-starts as a background service on Windows after install.
# Verify it is running: curl http://localhost:11434

# 6. Add your PDFs
# Place these 5 files into data/pdfs/
# (exact filenames must match config.py DOCUMENT_COMPANY_MAP):
#
#   TCPL_Q4FY26_press_release.pdf
#   tata-power-pr-q1fy26.pdf
#   SignedSEIntimation-Presentation_q1fy26_Voltaz.pdf
#   press-release-and-analyst-presentation_Tata_Steel.pdf
#   Earnings-call-transcript-Q1FY26_Tata_Chemical.pdf

# 7. Build the FAISS vector index (run once, or whenever PDFs change)
# Embeddings run locally via HuggingFace — no API key needed for this step.
python src/ingest.py

# 8. Launch the app
streamlit run src/app.py
```

The app opens at `http://localhost:8501`.

---

## 4. Demo Users (simulated auth)

| Email | Authorized Companies |
|---|---|
| alice@email.com | TCPL |
| bob@email.com | TataPower, Voltas |
| charlie@email.com | TataSteel, TataChemicals |
| admin@email.com | All 5 companies |

Edit `src/config.py` → `USER_ACCESS_MAP` to add or change users.

---

## 5. Demonstrating the Assignment Requirements

### Test 1 — Login & Access Scope
Log in as each demo user. Confirm the sidebar shows the correct authorized
companies for each user. Demonstrates simulated authentication and access mapping.

### Test 2 — Query Isolation (core requirement)
Log in as `alice@email.com`. Ask:
> *"What is the revenue for Tata Steel?"*

Expected: system responds that Tata Steel is not within Alice's authorized
access — no Tata Steel data is ever retrieved because the FAISS filter
blocks it structurally before retrieval runs.

Log in as `charlie@email.com` and ask the same question.
Expected: real Tata Steel figures, correctly retrieved and attributed.

For an objective, automated proof of isolation, run:
```bash
python src/test_isolation.py
```
This sends the same probe query as every registered user and prints a
PASS/FAIL report confirming no unauthorized chunks were retrieved.

### Test 3 — Conversational Context (follow-up questions)
Log in as `admin@email.com`. Ask:
1. *"What was the PAT for TCPL in the most recent quarter?"*
2. *"How does that compare to the full year?"*
3. *"What drove that growth?"*

Each follow-up is answered in context of the previous turns. The
`history_aware_retriever` rewrites follow-up questions into standalone
queries using that user's `chat_history` before hitting FAISS.

### Test 4 — Multi-User Session Isolation
Open two browser tabs simultaneously. Log in as different users in each.
Ask different questions. Reset memory in one tab — the other tab's history
is unaffected. Each `UserSession` is keyed by email server-side in
`qa_engine._active_sessions`.

### Test 5 — Provider Fallback Resilience
With Groq as primary, temporarily disconnect from the internet (or set an
invalid `GROQ_API_KEY` in `.env`). Ask any question. Expected: the system
silently falls back to Ollama locally and shows a calm blue info banner —
no crash, no stack trace. Restore the key and responses return to Groq.

---

## 6. Project Structure

```
doc-qa-system/
├── .env                    # YOUR real keys — git-ignored, never committed
├── .env.example            # key name template — safe to commit
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   └── pdfs/               # place the 5 source PDFs here
├── vectorstore/
│   └── faiss_index/        # created by ingest.py — git-ignored
│       ├── index.faiss
│       └── index.pkl
└── src/
    ├── config.py            # all settings: providers, users, access map
    ├── ingest.py            # PDF → chunk → tag → embed → FAISS index
    ├── access_control.py    # per-user company filter (single source of truth)
    ├── qa_engine.py         # RAG chains + per-user memory + fallback logic
    ├── app.py               # Streamlit UI
    └── test_isolation.py    # automated isolation proof script
```

---

## 7. Key Design Decisions

**Why a single FAISS index with metadata filters, not separate indexes per
company?**
A single shared index with row-level metadata security is the standard
production pattern — it is how Azure AI Search, Pinecone, and Weaviate
implement multi-tenant RAG. Separate indexes per company would be
operationally expensive (N indexes to maintain, update, and query) and
harder to extend. Changing a user's access permissions here is a one-line
change to `USER_ACCESS_MAP` in `config.py`.

**Why Groq as primary LLM?**
Groq's free tier provides access to Llama 3.3 70B at very fast inference
speeds (purpose-built inference hardware), with no billing requirement.
For a financial Q&A demo requiring reliable instruction-following and
accurate company-name attribution, a 70B model outperforms smaller local
models significantly.

**Why Ollama as fallback?**
Provides a fully offline safety net that requires no API key or internet
connection. If the cloud provider hits a rate limit during a live demo,
the system continues serving answers from a local model automatically.

**Why HuggingFace embeddings locally?**
Keeps the ingestion pipeline completely offline and free — no API key
required to build or rebuild the FAISS index. The `all-MiniLM-L6-v2`
model is small (~80MB), fast, and sufficient for semantic search over
financial document chunks.

**Extending to real authentication:**
Swap `USER_ACCESS_MAP` in `config.py` for a DB-backed lookup or identity
provider claims (e.g. Azure Entra ID group membership, AWS Cognito
attributes). `access_control.py` is the only file that would need to
change — the retrieval, memory, and UI layers are all auth-provider agnostic.