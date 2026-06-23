"""
config.py
---------
Central configuration for the Multi-User Document Search & Conversational
Q&A system. Keeping everything here means the rest of the codebase has
zero hardcoded paths/model names, which makes the system easy to demo
and easy to explain in an interview.
"""

import os
from dotenv import load_dotenv

# Loads variables from a local .env file (if present) into os.environ.
# .env is git-ignored — this is how GOOGLE_API_KEY reaches the app without
# ever being hardcoded or committed to the repo.
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss_index")

# ---------------------------------------------------------------------------
# Embeddings — always local (no API key required)
# ---------------------------------------------------------------------------
# Runs locally via sentence-transformers (HuggingFace).
# all-MiniLM-L6-v2 is small, fast, and good enough for semantic search demos.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# LLM Inference — pluggable provider
# ---------------------------------------------------------------------------
# Set to "ollama" (fully local, no key) or "google" (Gemini via Google AI
# Studio API key, free tier). qa_engine.py reads this one flag and builds
# the right LangChain chat model — nothing else in the codebase changes.
# ---------------------------------------------------------------------------
# LLM Inference — pluggable provider
# ---------------------------------------------------------------------------
# "groq"   → Groq cloud API (free tier, fast, recommended for demo)
# "ollama" → fully local, no key needed
# "google" → Gemini via Google AI Studio
LLM_PROVIDER = "groq"

# --- Groq settings (used when LLM_PROVIDER == "groq") ---
# Free API key at console.groq.com — no billing required.
# llama-3.3-70b-versatile: Groq's best free model, strong instruction-following.
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- Ollama settings (fallback — local, no key needed) ---
# Pull with: ollama pull phi3
OLLAMA_MODEL_NAME = "phi3"
OLLAMA_BASE_URL = "http://localhost:11434"

# --- Google AI Studio settings (optional third provider) ---
GOOGLE_MODEL_NAME = "gemini-2.0-flash"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# --- Automatic fallback ---
# On any quota/rate-limit/transient error from the primary provider,
# the system automatically retries against the fallback provider.
# Primary: groq | Fallback: ollama
ENABLE_LLM_FALLBACK = True
FALLBACK_PROVIDER = "ollama"  # explicit — don't infer from LLM_PROVIDER

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
# 6 (not 4): earnings-release PDFs are dense — the one sentence containing
# a specific metric (e.g. "Group Net Profit") often sits in a different
# chunk than the surrounding narrative text the query semantically matches
# best. A slightly wider k improves recall for these short, fact-dense
# pages at minimal extra cost.
TOP_K = 6

# ---------------------------------------------------------------------------
# Document -> Company mapping
# ---------------------------------------------------------------------------
# This is the metadata tag stamped onto every chunk from a given PDF.
# It's what the access-control filter matches against at query time.
DOCUMENT_COMPANY_MAP = {
    "TCPL_Q4FY26_press_release.pdf": "TCPL",
    "tata-power-pr-q1fy26.pdf": "TataPower",
    "SignedSEIntimation-Presentation_q1fy26_Voltaz.pdf": "Voltas",
    "press-release-and-analyst-presentation_Tata_Steel.pdf": "TataSteel",
    "Earnings-call-transcript-Q1FY26_Tata_Chemical.pdf": "TataChemicals",
}

# ---------------------------------------------------------------------------
# User -> Access Control mapping
# ---------------------------------------------------------------------------
# Dummy auth: email -> list of companies the user is allowed to query.
# In a real system this would live in a DB / IAM provider; here it's a
# simple dict to satisfy the assignment's "simulate access control" ask.
USER_ACCESS_MAP = {
    "alice@email.com": ["TCPL"],
    "bob@email.com": ["TataPower", "Voltas"],
    "charlie@email.com": ["TataSteel", "TataChemicals"],
    "admin@email.com": ["TCPL", "TataPower", "Voltas", "TataSteel", "TataChemicals"],
}