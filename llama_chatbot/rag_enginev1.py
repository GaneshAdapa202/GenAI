import os
from pathlib import Path
from dotenv import load_dotenv
 
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
import faiss
 
 
# ---------------- CONFIG ----------------
DATA_DIR = Path("uploaded_files")
DATA_DIR.mkdir(exist_ok=True)
 
FAISS_INDEX_PATH = Path("faiss.index")
 
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("⚠️ GOOGLE_API_KEY missing in .env")
 
# Gemini models
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash-lite")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/embedding-001")
 
 
class RAGEngine:
    def __init__(self):
        # LLM + Embeddings
        self.llm = GoogleGenAI(model=GEN_MODEL, api_key=API_KEY)
        self.embed_model = GoogleGenAIEmbedding(model_name=EMBED_MODEL, api_key=API_KEY)
        self.index = None
        self.faiss_index = None
        self.storage_context = None
 
    # ----------- Build Index -----------
    def build_rag_index(self, file_paths):
        # 1. Read docs
        docs = SimpleDirectoryReader(input_files=file_paths).load_data()
 
        # 2. Chunking
        splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
        nodes = splitter.get_nodes_from_documents(docs)
 
        # 3. FAISS setup → dynamically detect dimension
        test_emb = self.embed_model.get_text_embedding("test")
        dim = len(test_emb)
 
        self.faiss_index = faiss.IndexFlatL2(dim)
        vector_store = FaissVectorStore(faiss_index=self.faiss_index)
 
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
 
        # 4. Build vector index
        self.index = VectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
        )
 
        # Optionally save FAISS to disk
        faiss.write_index(self.faiss_index, str(FAISS_INDEX_PATH))
        return self.index
 
    # ----------- Query -----------
    def run_rag_pipeline(self, question: str, top_k: int = 5):
        """Return both response and retrieved chunks for evaluation"""
        if not self.index:
            raise RuntimeError("❌ Index not built yet. Run build_index first.")
 
        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            llm=self.llm,
        )
       
        # Fetch response
        response_obj = query_engine.query(question)
        response_text = str(response_obj)
 
        # Access retrieved chunks if available
        retrieved_chunks = []
        if hasattr(response_obj, "source_nodes"):
            retrieved_chunks = [n.node.get_text() for n in response_obj.source_nodes]
 
        return response_text, retrieved_chunks
 