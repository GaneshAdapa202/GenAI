"""
agent.py
A simple "agent" that:
- uses Tavily (tools.tavily_search) to find job pages
- uses Google GenAI (Gemini) to create embeddings and summaries
- stores jobs in SQLite (db.py)
"""

import os
import time
from typing import List, Dict, Optional
import numpy as np

from tools import tavily_search, basic_extract_job_from_result
import db

# Import the Google GenAI SDK
from google import genai

# Initialize Gemini client helper
def create_gemini_client(api_key: Optional[str] = None):
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in environment")
    client = genai.Client(api_key=api_key)
    return client

def embed_texts(client, texts):
    embeddings = []
    model = os.getenv("EMBED_MODEL", "models/text-embedding-004")  # ✅ safe default
    for text in texts:
        resp = client.models.embed_content(
            model=model,
            contents=text
        )
        vector = resp.embeddings[0].values
        embeddings.append(vector)
    return embeddings
def cosine_sim(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

class SimpleAgent:
    def __init__(self, db_path: str, gemini_client=None):
        self.db_path = db_path
        db.init_db(self.db_path)
        self.client = gemini_client or create_gemini_client()

    def ingest_from_web(self, keywords: str, location: Optional[str] = None, max_results: int = 5):
        """
        Search Tavily for "<keywords> jobs <location>" and ingest top results into DB.
        Precomputes embeddings using Gemini.
        """
        query = f"{keywords} jobs"
        if location:
            query += f" {location}"
        print(f"[agent] Searching web for: {query}")
        results = tavily_search(query, max_results=max_results)
        if not results:
            print("[agent] No results from Tavily.")
            return 0

        # Build job dicts and collect descriptions to embed
        jobs_to_add = []
        descriptions = []
        for r in results:
            job = basic_extract_job_from_result(r, default_location=location)
            print(job)
            # use snippet as description
            descriptions.append(job.get("description") or job.get("title") or "")
            jobs_to_add.append(job)

        # Create embeddings in one call (small batches)
        print("[agent] Creating embeddings with Gemini...")
        embeddings = embed_texts(self.client, descriptions)

        inserted = 0
        for job, emb in zip(jobs_to_add, embeddings):
            job["embedding"] = emb
            ok = db.insert_job(self.db_path, job)
            if ok:
                inserted += 1
        print(f"[agent] Inserted {inserted} new jobs into DB.")
        return inserted

    def semantic_search(self, natural_query: str, top_k: int = 5):
        """
        Embed the natural_query and compute cosine similarity against stored job embeddings.
        Returns top_k job dicts.
        """
        print("[agent] Creating embedding for query...")
        q_emb = embed_texts(self.client, [natural_query])[0]
        rows = db.get_all_jobs_with_embeddings(self.db_path)
        scored = []
        for r in rows:
            emb = r.get("embedding")
            if emb:
                score = cosine_sim(q_emb, emb)
                scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for s, r in scored[:top_k]]

    def answer_query_with_rag(self, natural_query: str, top_k: int = 3, model: str = "gemini-2.5-flash"):
        """
        Do a semantic search, then call Gemini to produce a short answer / summary
        using the matched job descriptions (RAG).
        """
        matches = self.semantic_search(natural_query, top_k=top_k)
        if not matches:
            return "No matching jobs found.", []

        # Build a prompt that includes matched job descriptions (keep it short)
        context_texts = []
        for m in matches:
            t = f"Title: {m.get('title')}\nCompany: {m.get('company')}\nLocation: {m.get('location')}\nUrl: {m.get('url')}\nDescription: {m.get('description')}\n"
            context_texts.append(t)

        # Create the RAG prompt (concatenate contexts)
        context_block = "\n---\n".join(context_texts)
        prompt = f"""
You are a helpful assistant specialized in summarizing job postings.
Context (top {top_k} matches):
{context_block}

User question: {natural_query}

Please summarize the most relevant job(s) for the user (title, company, location, short reason why match, and the url). Keep it concise.
"""
        print("[agent] Asking Gemini to produce final answer (RAG) ...")
        resp = self.client.models.generate_content(model=model, contents=prompt)
        # resp.text or resp.output_text depending on sdk — try common attr
        if hasattr(resp, "text"):
            answer = resp.text
        else:
            try:
                answer = resp["text"]
            except Exception:
                answer = str(resp)
        return answer.strip(), matches
