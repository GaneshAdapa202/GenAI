import os
os.environ["USE_TF"] = "0"
 
from dotenv import load_dotenv
load_dotenv()
 
import streamlit as st
from io import BytesIO
import json
from pathlib import Path
from typing import List, Dict, Tuple
import re
 
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import google.generativeai as genai
 
# Optional stronger reranker
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except Exception:
    CROSS_ENCODER_AVAILABLE = False
 
# Optional TF-IDF lexical fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
 
# ---------------- CONFIG ----------------
SAVE_DIR = Path("uploaded_files")
SAVE_DIR.mkdir(exist_ok=True)
META_PATH = Path("faiss_meta.json")
FAISS_INDEX_PATH = Path("faiss.index")
CHAT_HISTORY_PATH = Path("chat_history.json")
 
# Embedding model (auto-detect dim)
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-mpnet-base-v2")  # accurate
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-1.5-flash")
 
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
else:
    st.sidebar.warning("⚠️ GEMINI_API_KEY not set. Set it in your .env file.")
 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 32
 
# ---------------- Helpers: text extraction ----------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception:
        return ""
 
def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        doc = docx.Document(BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""
 
def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return BytesIO(file_bytes).read().decode("utf-8", errors="ignore")
    except Exception:
        return ""
 
# ---------------- Chunking ----------------
def heading_chunk_text(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n(?=\d+\.\s+)", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        return smart_chunk_text(text)
    out = []
    for p in parts:
        if len(p) <= CHUNK_SIZE:
            out.append(p)
        else:
            out.extend(smart_chunk_text(p))
    return out
 
def smart_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.replace("\r", " ")
    parts = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) > chunk_size * 1.5:
            subs = [s.strip() + ("." if not s.strip().endswith(".") else "")
                    for s in line.split(". ") if s.strip()]
            parts.extend(subs)
        else:
            parts.append(line)
 
    chunks, cur = [], ""
    for p in parts:
        if not cur:
            cur = p
        elif len(cur) + 1 + len(p) <= chunk_size:
            cur = (cur + " " + p).strip()
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks
 
# ---------------- Embeddings & FAISS ----------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)
 
embedder = load_embedder()
EMBED_DIM = embedder.get_sentence_embedding_dimension()
 
def normalize_embeddings(arr: np.ndarray):
    faiss.normalize_L2(arr)
    return arr
 
def create_faiss_index(dim: int):
    return faiss.IndexFlatIP(dim)
 
# ---------------- Persistence ----------------
def save_meta(meta_list: List[Dict]):
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)
 
def load_meta() -> List[Dict]:
    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []
 
def save_chat_history(history: List[Dict]):
    with open(CHAT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
 
def load_chat_history() -> List[Dict]:
    if CHAT_HISTORY_PATH.exists():
        with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []
 
# ---------------- Indexing ----------------
def build_index_from_documents(docs: List[Tuple[str, Dict]], progress_callback=None):
    doc_chunks = []
    for text, meta in docs:
        chunks = heading_chunk_text(text)
        for c in chunks:
            doc_chunks.append({"text": c, "meta": meta})
    total = len(doc_chunks)
    if total == 0:
        return None, []
    index = create_faiss_index(EMBED_DIM)
    meta_list = []
    for i in range(0, total, BATCH_SIZE):
        batch = doc_chunks[i:i+BATCH_SIZE]
        texts = [b["text"] for b in batch]
        embs = embedder.encode(texts, convert_to_numpy=True).astype("float32")
        normalize_embeddings(embs)
        index.add(embs)
        for b in batch:
            meta_list.append({"source": b["meta"].get("source"), "text": b["text"]})
        if progress_callback:
            progress_callback(min(1.0, (i + BATCH_SIZE) / total))
    return index, meta_list
 
# ---------------- Retrieval ----------------
def overfetch_faiss(index, meta_list, query: str, fetch_k: int = 20):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    normalize_embeddings(q_emb)
    D, I = index.search(q_emb, fetch_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx != -1 and idx < len(meta_list):
            results.append({
                "score": float(score),
                "source": meta_list[idx]["source"],
                "text": meta_list[idx]["text"]
            })
    return results
 
@st.cache_resource
def load_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    if not CROSS_ENCODER_AVAILABLE:
        return None
    return CrossEncoder(model_name)
 
def rerank_with_tfidf(retrieved, query: str):
    if not retrieved:
        return []
    texts = [r["text"] for r in retrieved]
    if SKLEARN_AVAILABLE:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
        doc_vectors = vectorizer.fit_transform([query] + texts)
        qv = doc_vectors[0]
        tv = doc_vectors[1:]
        tfidf_scores = cosine_similarity(qv, tv)[0]
        for r, s in zip(retrieved, tfidf_scores):
            r["tfidf_score"] = float(s)
        return sorted(retrieved, key=lambda x: x.get("tfidf_score", 0.0), reverse=True)
    return retrieved
 
# ---------------- Context + Generation ----------------
def prepare_context_and_quotes(ranked, query, top_k_context: int = 1, max_chars: int = 4000):
    selected = ranked[:top_k_context]
    ctx = ""
    for r in selected:
        block = f"Source: {r['source']}\n{r['text']}\n\n"
        if len(ctx) + len(block) > max_chars:
            break
        ctx += block
    return ctx.strip(), selected
 
def generate_with_gemini(query: str, context: str):
    if not GEMINI_KEY:
        raise RuntimeError("Gemini API key not set")
    model = genai.GenerativeModel(GEN_MODEL)
    prompt = f"""You are a careful assistant. Use ONLY the context provided.
 
Question:
{query}
 
Context:
{context}
 
If context is insufficient, reply "I don't know".
"""
    resp = model.generate_content(prompt)
    return resp.text
 
# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("💬 RAG Chat Assistant")
 
# Sidebar: Upload files
st.sidebar.header("📂 Upload Documents")
if "documents" not in st.session_state:
    st.session_state.documents = []
 
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / DOCX / TXT", accept_multiple_files=True,
    type=["pdf", "docx", "txt"]
)
 
if uploaded_files:
    for f in uploaded_files:
        if any(doc[1]["source"] == f.name for doc in st.session_state.documents):
            continue
        if f.name.lower().endswith(".pdf"):
            text = extract_text_from_pdf(f.getvalue())
        elif f.name.lower().endswith(".docx"):
            text = extract_text_from_docx(f.getvalue())
        else:
            text = extract_text_from_txt(f.getvalue())
        if text.strip():
            st.session_state.documents.append((text, {"source": f.name}))
            st.sidebar.success(f"Processed {f.name}")
        else:
            st.sidebar.warning(f"No text extracted from {f.name}")
 
# Build index button (only before processing)
if st.session_state.documents and "faiss_index" not in st.session_state:
    if st.sidebar.button("Build RAG Index"):
        progress_bar = st.sidebar.progress(0.0)
        idx, meta_list = build_index_from_documents(
            st.session_state.documents, progress_callback=lambda f: progress_bar.progress(f)
        )
        progress_bar.empty()
        if idx:
            st.session_state.faiss_index = idx
            st.session_state.meta_list = meta_list
            faiss.write_index(idx, str(FAISS_INDEX_PATH))
            save_meta(meta_list)
            st.sidebar.success(f"Indexed {len(meta_list)} chunks.")
 
# Sidebar: Old chats
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
st.sidebar.markdown("---")
st.sidebar.header("💬 Previous Queries")
for chat in reversed(st.session_state.chat_history[-50:]):
    st.sidebar.markdown(f"- {chat['query']}")
 
# Main chat area
st.header("Ask a question")
query = st.text_input("Your question")
 
if st.button("Send") and query.strip():
    if "faiss_index" not in st.session_state:
        st.warning("No index found. Upload files and build index first.")
    else:
        candidates = overfetch_faiss(st.session_state.faiss_index, st.session_state.meta_list, query)
        if not candidates:
            st.warning("No candidate passages found.")
        else:
            ranked = rerank_with_tfidf(candidates, query)
            context, selected = prepare_context_and_quotes(ranked, query)
            try:
                with st.spinner("Generating answer..."):
                    answer = generate_with_gemini(query, context)
            except Exception as e:
                answer = f"Generation failed: {e}"
 
            st.session_state.chat_history.append({
                "query": query,
                "answer": answer
            })
            save_chat_history(st.session_state.chat_history)
 
            st.subheader("Assistant — Answer")
            st.write(answer)