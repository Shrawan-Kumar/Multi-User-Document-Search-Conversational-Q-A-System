"""
ingest.py
---------
Loads all PDFs from data/pdfs, splits them into chunks, tags every chunk
with a `company` metadata field (used later for access-control filtering),
embeds them locally via sentence-transformers, and persists a FAISS index
to disk.

Run this once (or whenever PDFs change):
    python src/ingest.py
"""

import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


def load_and_tag_documents():
    """Load every PDF in PDF_DIR and stamp each page with company metadata."""
    all_docs = []

    for filename, company in config.DOCUMENT_COMPANY_MAP.items():
        filepath = os.path.join(config.PDF_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  [WARN] Missing file, skipping: {filename}")
            continue

        print(f"  Loading {filename}  -> tagged as company='{company}'")
        loader = PyPDFLoader(filepath)
        pages = loader.load()

        for page in pages:
            page.metadata["company"] = company
            page.metadata["source_file"] = filename

        all_docs.extend(pages)

    return all_docs


def chunk_documents(documents):
    """Split loaded pages into overlapping chunks for better retrieval granularity."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def build_vectorstore(chunks):
    """Embed chunks locally and persist a FAISS index to disk."""
    print(f"\nEmbedding {len(chunks)} chunks with '{config.EMBEDDING_MODEL_NAME}' "
          f"(runs locally, first call downloads the model)...")

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(config.VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(config.FAISS_INDEX_PATH)
    print(f"FAISS index saved to: {config.FAISS_INDEX_PATH}")
    return vectorstore


def main():
    print("=" * 70)
    print("INGESTION: Multi-User Document Search & Conversational Q&A System")
    print("=" * 70)

    print("\n[1/3] Loading & tagging PDFs by company...")
    documents = load_and_tag_documents()
    if not documents:
        print("\nNo PDFs found! Place your 5 Tata company PDFs into:")
        print(f"  {config.PDF_DIR}")
        print("Expected filenames (see config.DOCUMENT_COMPANY_MAP):")
        for f in config.DOCUMENT_COMPANY_MAP:
            print(f"  - {f}")
        return
    print(f"  Loaded {len(documents)} pages total.")

    print("\n[2/3] Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks "
          f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).")

    print("\n[3/3] Building FAISS vector store...")
    build_vectorstore(chunks)

    print("\nDone. You can now run the app with:")
    print("  streamlit run src/app.py")


if __name__ == "__main__":
    main()