# app.py (project root)
"""
Aksum Procurement RAG Assistant - Main Streamlit Application
Enhanced with authentication, logging, and LLM generation
"""

import streamlit as st

# --- MUST be the first Streamlit call ---
st.set_page_config(
    page_title="Aksum Procurement Assistant",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Standard libs ---
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# --- Path setup ---
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Local imports ---
try:
    import src.rag_chat as rag_chat
    from src.embed_store import discover_and_chunk, build_index, save_artifacts
    from src.generate import generate_answer
except ImportError:
    # Fallback if run differently
    import rag_chat
    from embed_store import discover_and_chunk, build_index, save_artifacts

    try:
        from generate import generate_answer
    except ImportError:
        def generate_answer(query, sources, extractive_answer):
            return extractive_answer, False

# --- Authentication Layer (optional) ---
def check_authentication():
    """Handle simple password-based authentication if configured"""
    secret_key = os.getenv("AKSUM_CHAT_SECRET", "").strip()

    if not secret_key:
        return True  # No auth required

    if "auth_verified" not in st.session_state:
        st.session_state["auth_verified"] = False

    if not st.session_state["auth_verified"]:
        st.sidebar.markdown("### 🔐 Access Control")
        entered_key = st.sidebar.text_input("Enter access key:", type="password")

        if st.sidebar.button("Authenticate", key="auth_btn"):
            if entered_key.strip() == secret_key:
                st.session_state["auth_verified"] = True
                st.sidebar.success("Access granted!")
                st.rerun()
            else:
                st.sidebar.error("Invalid access key")

        if not st.session_state["auth_verified"]:
            st.warning("🔒 Authentication required to access the assistant.")
            st.stop()

    return True

# Run auth first
check_authentication()

# --- Main UI header ---
st.title("📦 Aksum RAG Procurement Assistant")
st.caption("Internal AI assistant • Document retrieval with intelligent answers • Secure and logged")

# --- Logging ---
def log_query_interaction(query: str, used_llm: bool, sources: list, response_length: int = 0):
    """Log query interactions for analysis and monitoring"""
    try:
        logs_directory = Path("logs")
        logs_directory.mkdir(exist_ok=True)

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "llm_enhanced": used_llm,
            "source_count": len(sources),
            "response_length": response_length,
            "sources_metadata": [
                {
                    "doc_id": src.get("id", src.get("chunk_id", "unknown")),
                    "relevance_score": round(src.get("score", 0.0), 3),
                    "source_path": src.get("source", "unknown"),
                    "title": src.get("title", "untitled")
                }
                for src in sources[:5]
            ]
        }

        log_file = logs_directory / "interaction_log.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"[app] Failed to log interaction: {e}")

# --- Sidebar: Document Management ---
with st.sidebar:
    st.markdown("### 📁 Document Management")

    # Upload new files
    st.caption("Upload .md, .txt, or .csv files")
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["md", "txt", "csv"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_files:
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []
        for uf in uploaded_files:
            out_path = upload_dir / uf.name
            try:
                with open(out_path, "wb") as f:
                    f.write(uf.getbuffer())
                saved_files.append(uf.name)
            except Exception as e:
                st.error(f"❌ Failed to save {uf.name}: {e}")

        if saved_files:
            st.success(f"Saved {len(saved_files)} file(s)")
            with st.expander("Uploaded files"):
                for fname in saved_files:
                    st.text(f"• {fname}")
            st.info("Rebuild the index to include them in search results.")

    st.markdown("---")

    # Rebuild index
    if st.button("🔄 Rebuild Search Index", key="rebuild_btn"):
        with st.spinner("Rebuilding search index..."):
            try:
                chunks = discover_and_chunk("data")
                if not chunks:
                    st.error("⚠️ No documents found. Add files under `data/` first.")
                else:
                    idx, embs = build_index(chunks)
                    model_id = "sentence-transformers/all-MiniLM-L6-v2"
                    save_artifacts(idx, chunks, embs, model_id)
                    st.success(f"✅ Index rebuilt with {len(chunks)} chunks.")
                    print(f"[app] Index rebuild complete: {len(chunks)} chunks")
            except Exception as e:
                st.error(f"❌ Index rebuild failed: {e}")
                print(f"[app] Rebuild error: {e}")

# --- Query input ---
st.markdown("### 💬 Ask Questions About Your Documents")
user_query = st.text_input(
    "Your question:",
    placeholder="e.g., What are the penalty terms for late delivery of TMT bars?",
    key="query_input"
)

search_initiated = st.button("🔍 Search", key="search_btn", disabled=not user_query.strip())

with st.expander("💡 Tips", expanded=False):
    st.markdown("""
    - Be specific: product codes, categories, supplier names
    - Upload new docs & rebuild index before asking
    - If no results, try rephrasing
    """)

# --- Main query handling ---
if search_initiated:
    if not user_query.strip():
        st.warning("⚠️ Please enter a question first.")
    else:
        try:
            print(f"[app] Query: {user_query}")
            docs = rag_chat.retrieve(user_query, top_k=5)

            if not docs:
                st.info("🔍 No documents found. Try rephrasing or rebuild the index.")
            else:
                extractive, cited = rag_chat.build_answer(user_query, docs)
                final_text, used_llm = generate_answer(user_query, cited, extractive)

                colA, colB = st.columns([0.65, 0.35], gap="large")

                with colA:
                    st.subheader("📝 Answer")
                    st.write(final_text)
                    st.caption("🤖 LLM pin-point (+ citations)" if used_llm else "📄 Direct extractive answer")

                with colB:
                    st.subheader("📚 Sources")
                    if not cited:
                        st.info("No sources available")
                    else:
                        for i, h in enumerate(cited[:3], 1):
                            st.markdown(f"**[{i}] {h['title']}**  \n`{h['source']}`")
                            st.caption(h.get("text_preview", "No preview"))

                log_query_interaction(
                    query=user_query,
                    used_llm=used_llm,
                    sources=cited,
                    response_length=len(final_text)
                )

        except FileNotFoundError:
            st.error("❌ Search index not found. Please rebuild from the sidebar.")
        except Exception as e:
            st.error(f"❌ Error processing query: {e}")
            print(f"[app] Error: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Aksum Procurement Assistant v1.0 • Internal prototype")

if os.getenv("STREAMLIT_ENV") == "development":
    with st.expander("🔧 Debug Info"):
        st.json({
            "auth_enabled": bool(os.getenv("AKSUM_CHAT_SECRET")),
            "logs_directory": str(Path('logs').absolute()),
            "data_directory": str(Path('data').absolute())
        })
