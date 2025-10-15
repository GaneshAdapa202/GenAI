import os
import shutil
import tempfile
from pathlib import Path

import gradio as gr

from rag_engine import RAGEngine
from evaluator import evaluate_response

# ---------------- Setup ----------------
DATA_DIR = Path("uploaded_files")
DATA_DIR.mkdir(exist_ok=True)

rag_engine = RAGEngine()

# ---------------- Helper Functions ----------------
def _save_uploaded_files(uploaded_files):
    if not uploaded_files:
        return []
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    for f in uploaded_files:
        dest = os.path.join(temp_dir, os.path.basename(f.name))
        shutil.copy(f.name, dest)
        file_paths.append(dest)
    return file_paths

def build_index(uploaded_files):
    if not uploaded_files:
        return "⚠️ No files uploaded. Please upload PDF/DOCX/TXT files.", [], gr.update(visible=False)

    file_paths = _save_uploaded_files(uploaded_files)

    try:
        idx = rag_engine.build_rag_index(file_paths)
        rag_engine.index = idx
    except Exception as e:
        return f"❌ Failed to build index: {e}", [], gr.update(visible=False)

    return "✅ Index built! You can now start chatting.", [], gr.update(visible=True)

def submit_query(user_input, history):
    if not user_input:
        return history, gr.update(value=""), gr.update(visible=False), ""

    if not getattr(rag_engine, "index", None):
        bot_reply = "❌ Index not built. Please upload documents and click 'Build RAG Index'."
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": bot_reply})
        return history, gr.update(value=""), gr.update(visible=False), ""

    try:
        response_text, retrieved_chunks = rag_engine.run_rag_pipeline(user_input)
    except Exception as e:
        response_text = f"❌ Error running RAG pipeline: {e}"
        retrieved_chunks = []

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response_text})

    rag_engine._latest_answer = response_text
    rag_engine._latest_chunks = retrieved_chunks

    eval_visible = True if retrieved_chunks else False
    return history, gr.update(value=""), gr.update(visible=eval_visible), ""

def evaluate_latest():
    latest_answer = getattr(rag_engine, "_latest_answer", None)
    latest_chunks = getattr(rag_engine, "_latest_chunks", None)

    if not latest_answer or not latest_chunks:
        return "⚠️ No answer available to evaluate. Ask a question first.", ""

    try:
        scores = evaluate_response(latest_answer, latest_chunks)
    except Exception as e:
        return f"❌ Evaluation failed: {e}", ""

    md = "### 📈 Evaluation Metrics\n\n"
    md += f"- **Relevance of Most Relevant Chunk (Max Cosine):** {scores.get('max_cosine', 0):.3f}  \n"
    md += f"- **Overall Relevance (Avg Cosine):** {scores.get('avg_cosine', 0):.3f}  \n"
    md += f"- **Max Euclidean (1/(1+d)):** {scores.get('max_euclidean', 0):.3f}  \n"
    md += f"- **Avg Euclidean (1/(1+d)):** {scores.get('avg_euclidean', 0):.3f}  \n"
    md += f"- **Max Dot:** {scores.get('max_dot', 0):.3f}  \n"
    md += f"- **Avg Dot:** {scores.get('avg_dot', 0):.3f}  \n"
    md += f"- **Max Manhattan (1/(1+L1)):** {scores.get('max_manhattan', 0):.3f}  \n"
    md += f"- **Avg Manhattan (1/(1+L1)):** {scores.get('avg_manhattan', 0):.3f}  \n"
    md += f"- **Chunks Retrieved:** {scores.get('retrieved_chunks', 0)}  \n"

    chunks_md = ""
    for i, c in enumerate(latest_chunks[:5], start=1):
        preview = c if len(c) <= 1000 else c[:1000] + "..."
        chunks_md += f"**Chunk {i}:**\n```\n{preview}\n```\n\n"

    return md, chunks_md

# ---------------- Build Gradio Interface ----------------
def build_ui():
    with gr.Blocks(title="Smart Document Assistant (Gradio)", css=None, css_paths=None, js=None, head=None, head_paths=None) as demo:
        gr.Markdown("# 📄 Smart Document Assistant")
        gr.Markdown("Upload documents, ask questions, and get intelligent answers backed by your content.")

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("## Step 1: Upload Documents")
                file_input = gr.File(label="Upload Files (pdf, docx, txt)", file_count="multiple", file_types=[".pdf", ".docx", ".txt"])
                build_btn = gr.Button("🔍 Build RAG Index", variant="primary")
                status_output = gr.Markdown("No index built yet.", elem_id="build_status")

                gr.Markdown("---")

                chatbox = gr.Chatbot(value=[], type="messages", visible=False, label="Chat — Unsaved Chat")
                chat_history_state = gr.State(value=[])
                user_input = gr.Textbox(placeholder="💬 Type your question here...", lines=1)
                submit_btn = gr.Button("Send", variant="primary")

                gr.Markdown("---")
                gr.Markdown("## Step 3: Evaluate Answer")
                eval_btn = gr.Button("📊 Evaluate Answer", visible=False)
                eval_md = gr.Markdown("", visible=True)
                with gr.Accordion("View Retrieved Chunks", open=False):
                    chunks_md_box = gr.Markdown("")

            with gr.Column(scale=1):
                gr.Markdown("## 💬 Chat Manager")
                saved_chats_md = gr.Markdown("No saved chats yet.")

        build_btn.click(build_index, inputs=file_input, outputs=[status_output, chatbox, chatbox], show_progress=False)
        submit_btn.click(submit_query, inputs=[user_input, chat_history_state], outputs=[chatbox, user_input, eval_btn, status_output])
        user_input.submit(submit_query, inputs=[user_input, chat_history_state], outputs=[chatbox, user_input, eval_btn, status_output])
        eval_btn.click(evaluate_latest, inputs=[], outputs=[eval_md, chunks_md_box])

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", share=False)
