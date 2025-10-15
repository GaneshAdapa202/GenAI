import streamlit as st
import tempfile
import os
from rag_enginev1 import RAGEngine  # Updated RAGEngine
from evaluator import evaluate_response
 
# ---------------- UI Setup ----------------
st.set_page_config(page_title="Smart Document Assistant", layout="wide")
st.title("📄 Smart Document Assistant")
st.markdown(
    "Upload documents, ask questions, and get intelligent answers backed by your content."
)
 
 
rag_engine = RAGEngine()
 
# ---------------- Session State ----------------
for key, default in [("rag_index", None), ("chats", {}), ("current_chat", None),
                     ("messages", []), ("chat_enabled", False), ("latest_answer", None),
                     ("latest_chunks", []), ("eval_ready", False)]:
    if key not in st.session_state:
        st.session_state[key] = default
 
# ---------------- Sidebar: Chat Manager ----------------
st.sidebar.header("💬 Chat Manager")
if st.session_state.chats:
    selected_chat = st.sidebar.radio("Select Saved Chat:", list(st.session_state.chats.keys()))
    if selected_chat != st.session_state.current_chat:
        st.session_state.current_chat = selected_chat
        st.session_state.messages = st.session_state.chats[selected_chat].copy()
        st.session_state.chat_enabled = True
else:
    st.sidebar.info("No saved chats yet.")
 
chat_name = st.sidebar.text_input("Save Current Chat As:", "")
if st.sidebar.button("💾 Save Chat"):
    if chat_name.strip():
        st.session_state.chats[chat_name] = st.session_state.messages.copy()
        st.session_state.current_chat = chat_name
        st.success(f"Chat saved as '{chat_name}'")
        st.rerun()
    else:
        st.sidebar.error("Please enter a chat name before saving.")
 
# ---------------- Step 1: Upload Documents ----------------
st.markdown("## Step 1: Upload Documents")
st.markdown("Upload your PDFs, DOCX, or TXT files to build the RAG index.")
uploaded_files = st.file_uploader(
    "Upload Files...", type=["pdf", "docx", "txt"], accept_multiple_files=True
)
 
if uploaded_files:
    if st.button("🔍 Build RAG Index"):
        with st.spinner("Building RAG index..."):
            temp_dir = tempfile.mkdtemp()
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)
 
            idx = rag_engine.build_rag_index(file_paths)
            if idx:
                st.session_state.rag_index = idx
                rag_engine.index = idx
                st.session_state.chat_enabled = True
                st.session_state.current_chat = "Unsaved Chat"
                st.session_state.messages = []
                st.success("✅ Index built! You can now start chatting.")
 
# ---------------- Step 2: Ask Questions ----------------
if st.session_state.chat_enabled and st.session_state.current_chat:
    st.markdown(f"## Step 2: Chat — {st.session_state.current_chat}")
    st.markdown("Ask questions based on your uploaded documents.")
 
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            role_label = "You" if message["role"] == "user" else "RAG Answer"
            st.markdown(f"**{role_label}:** {message['content']}")
 
    # Chat input
    query = st.chat_input("💬 Type your question here...")
 
    if query and st.session_state.rag_index:
        rag_engine.index = st.session_state.rag_index
        response_text, retrieved_chunks = rag_engine.run_rag_pipeline(query)
 
        st.session_state.latest_answer = response_text
        st.session_state.latest_chunks = retrieved_chunks
        st.session_state.eval_ready = False
 
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "assistant", "content": response_text})
 
        # Display latest messages immediately
        with st.chat_message("user"):
            st.markdown(f"**You:** {query}")
        with st.chat_message("assistant"):
            st.markdown(f"**Bot Answer:** {response_text}")
 
# ---------------- Step 3: Evaluate Answer ----------------
if st.session_state.latest_answer and st.session_state.latest_chunks:
    st.markdown("---")
    st.markdown("## Step 3: Evaluate Answer")
    st.caption("Click the button to see how well the answer matches retrieved chunks.")
 
    if st.button("📊 Evaluate Answer"):
        st.session_state.eval_ready = True
 
    if st.session_state.eval_ready:
        eval_scores = evaluate_response(
            st.session_state.latest_answer,
            st.session_state.latest_chunks
        )
 
        st.subheader("📈 Evaluation Metrics")
        st.markdown(
            f"- **Relevance of Most Relevant Chunk (Max Cosine):** {eval_scores['max_cosine']:.3f}  \n"
            "  *How closely the answer matches the single most relevant piece of information.*"
        )
        st.markdown(
            f"- **Overall Relevance (Avg Cosine):** {eval_scores['avg_cosine']:.3f}  \n"
            "  *How well the answer aligns with all retrieved content on average.*"
        )
        st.markdown(
            f"- **Chunks Retrieved:** {eval_scores['retrieved_chunks']}  \n"
            "  *Number of document pieces used to generate the answer (more chunks = more context).*"
        )
 
        with st.expander("View Retrieved Chunks"):
            for i, chunk in enumerate(st.session_state.latest_chunks, 1):
                st.markdown(f"**Chunk {i}:** {chunk[:300]}...")