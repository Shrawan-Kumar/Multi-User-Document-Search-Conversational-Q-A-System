"""
app.py
------
Streamlit UI for the Multi-User Document Search & Conversational Q&A System.

Run with:
    streamlit run src/app.py
"""

import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from access_control import AccessDeniedError, list_registered_users
from qa_engine import get_or_create_session, clear_session

st.set_page_config(page_title="Multi-User Document Q&A", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Session state (this is Streamlit's per-browser-tab state — separate from
# our own UserSession objects in qa_engine, which are keyed by email and
# persist server-side across reruns / reconnects for the same user)
# ---------------------------------------------------------------------------
if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []  # for rendering chat bubbles only

# ---------------------------------------------------------------------------
# Sidebar: Login
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔐 Login")
    st.caption("Simulated auth — enter a registered demo email.")

    with st.expander("Registered demo users", expanded=False):
        for email in list_registered_users():
            companies = config.USER_ACCESS_MAP[email]
            st.markdown(f"**{email}**  \nAccess: {', '.join(companies)}")

    email_input = st.text_input("Email", placeholder="alice@email.com")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Log in", use_container_width=True):
            try:
                session = get_or_create_session(email_input)
                st.session_state.logged_in_user = session.user_email
                st.session_state.ui_messages = []
                st.success(f"Logged in as {session.user_email}")
                st.rerun()
            except AccessDeniedError as e:
                st.error(str(e))
            except FileNotFoundError as e:
                st.error(str(e))
    with col2:
        if st.button("Log out", use_container_width=True):
            if st.session_state.logged_in_user:
                clear_session(st.session_state.logged_in_user)
            st.session_state.logged_in_user = None
            st.session_state.ui_messages = []
            st.rerun()

    if st.session_state.logged_in_user:
        st.divider()
        allowed = config.USER_ACCESS_MAP[st.session_state.logged_in_user]
        st.info(f"**Current user:** {st.session_state.logged_in_user}\n\n"
                f"**Authorized companies:** {', '.join(allowed)}")

        session = get_or_create_session(st.session_state.logged_in_user)
        if config.ENABLE_LLM_FALLBACK:
            if session.fallback_available:
                st.caption(f"✅ Fallback armed: '{session.fallback_provider}' "
                           f"will be used automatically if '{session.primary_provider}' errors out.")
            else:
                st.caption(f"⚠️ Fallback to '{session.fallback_provider}' is enabled but "
                           f"unavailable right now (check it's installed/running).")

        if st.button("🔄 Reset conversation memory", use_container_width=True):
            session.reset_memory()
            st.session_state.ui_messages = []
            st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("📊 Multi-User Document Search & Conversational Q&A")
st.caption(f"LLM: {config.LLM_PROVIDER} (primary) "
           f"{'· auto-fallback to ' + config.FALLBACK_PROVIDER + ' on quota/errors' if config.ENABLE_LLM_FALLBACK else ''} "
           f"· Embeddings: local HuggingFace · Retrieval: FAISS")

if not st.session_state.logged_in_user:
    st.warning("👈 Please log in with a registered email to begin.")
    st.markdown("""
    **Try it out:**
    - `alice@email.com` → can only ask about **TCPL**
    - `bob@email.com` → can ask about **TataPower** and **Voltas**
    - `charlie@email.com` → can ask about **TataSteel** and **TataChemicals**
    - `admin@email.com` → can ask about all five companies

    Log in as two different users (e.g. in two browser tabs) to see that
    each user's query results and conversation memory stay fully isolated.
    """)
    st.stop()

# Render existing chat history
for msg in st.session_state.ui_messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("fell_back"):
            st.info(f"ℹ️ Served by local model ({msg.get('provider_used', 'ollama')}) — "
                    f"cloud provider was momentarily unavailable "
                    f"({msg.get('fallback_reason', 'transient error')}).")
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 Sources used ({len(msg['sources'])} chunks)"):
                for s in msg["sources"]:
                    st.markdown(
                        f"- **{s['company']}** — `{s['source_file']}` (page {s['page']})\n\n"
                        f"  > {s['snippet']}..."
                    )

# Chat input
query = st.chat_input("Ask a question about your authorized company documents...")

if query:
    st.session_state.ui_messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving access-filtered context and generating answer..."):
            session = get_or_create_session(st.session_state.logged_in_user)
            result = session.ask(query)
            if result.get("fell_back"):
                st.info(f"ℹ️ Served by local model ({result['provider_used']}) — "
                        f"cloud provider was momentarily unavailable "
                        f"({result.get('fallback_reason', 'transient error')}).")
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander(f"📎 Sources used ({len(result['sources'])} chunks)"):
                    for s in result["sources"]:
                        st.markdown(
                            f"- **{s['company']}** — `{s['source_file']}` (page {s['page']})\n\n"
                            f"  > {s['snippet']}..."
                        )

    st.session_state.ui_messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "fell_back": result.get("fell_back", False),
        "provider_used": result.get("provider_used"),
        "fallback_reason": result.get("fallback_reason"),
    })