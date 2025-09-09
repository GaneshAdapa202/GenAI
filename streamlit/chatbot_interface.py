import streamlit as st
import time

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG-powered Chatbot Interface")

# Session State Initialization
if "chats" not in st.session_state:
    st.session_state.chats = {}  # {chat_name: messages}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_enabled" not in st.session_state:
    st.session_state.chat_enabled = False
if "metrics_enabled" not in st.session_state:
    st.session_state.metrics_enabled = False
if "metrics" not in st.session_state:
    st.session_state.metrics = {}

# Sidebar: Saved Chats
st.sidebar.header("Controls")

if st.session_state.chats:
    selected_chat = st.sidebar.radio("Saved Chats", list(st.session_state.chats.keys()))
    if selected_chat != st.session_state.current_chat:
        st.session_state.current_chat = selected_chat
        st.session_state.messages = st.session_state.chats[selected_chat].copy()
        st.session_state.chat_enabled = True
else:
    st.sidebar.write("No chats saved yet.")

# Step 1: Upload PDFs
st.subheader("Step 1: Upload PDF Files")
uploaded_files = st.file_uploader("Upload PDFs...", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Build RAG Index"):
        with st.spinner("Building index..."):
            time.sleep(2)
        st.success("Index built! You can now chat.")

        st.session_state.chat_enabled = True  # Enable chat
        st.session_state.metrics_enabled = False  # Reset metrics until first query
        st.session_state.current_chat = "Unsaved Chat"
        st.session_state.messages = []

# Step 2: Chat with PDFs
if st.session_state.get("current_chat") and st.session_state.chat_enabled:
    st.subheader(f"Chat: {st.session_state.current_chat}")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input new query
    query = st.chat_input("Ask something...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Placeholder bot response
        response = f"Bot: Answer for '{query}'"
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        # ✅ Enable evaluation only after first query-response
        st.session_state.metrics_enabled = True

# Step 3: Evaluation (only after first chat)
if st.session_state.metrics_enabled:
    st.subheader("Step 3: Evaluation Metrics")

    if st.button("Run Evaluation"):
        with st.spinner("Computing metrics..."):
            time.sleep(2)  # simulate evaluation
            st.session_state.metrics = {
                "Accuracy": "85%",
                "Relevance": "90%",
                "Response Time": "1.2s"
            }
        st.success("Evaluation completed!")

    if st.session_state.metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", st.session_state.metrics["Accuracy"])
        col2.metric("Relevance", st.session_state.metrics["Relevance"])
        col3.metric("Response Time", st.session_state.metrics["Response Time"])

# Step 4: Save Chat
if st.session_state.get("current_chat"):
    st.subheader("Save Chat")
    chat_name = st.text_input("Enter a name for this chat:")

    if st.button("Save Chat"):
        if chat_name:
            st.session_state.chats[chat_name] = st.session_state.messages.copy()
            st.session_state.current_chat = chat_name
            st.success(f"Chat saved as '{chat_name}'")
            st.rerun()
        else:
            st.error("Please enter a chat name before saving.")
