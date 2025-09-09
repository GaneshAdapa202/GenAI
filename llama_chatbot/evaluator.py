from sentence_transformers import SentenceTransformer, util
import torch
 
# Load evaluation embedding model once
eval_model = SentenceTransformer("all-MiniLM-L6-v2")
 
def evaluate_response(answer, retrieved_chunks):
    """Compute multiple semantic similarity scores between answer and retrieved chunks"""
    if not retrieved_chunks:
        return {"error": "No chunks retrieved"}
 
    # Encode answer and chunks
    answer_emb = eval_model.encode(answer, convert_to_tensor=True)
    chunk_embs = eval_model.encode(retrieved_chunks, convert_to_tensor=True)
 
    # Cosine similarity
    cosine_scores = util.cos_sim(answer_emb, chunk_embs)
   
    # Euclidean similarity (1 / (1 + distance))
    euclidean_dist = torch.norm(chunk_embs - answer_emb, dim=1)
    euclidean_sim = 1 / (1 + euclidean_dist)
 
    # Dot product similarity
    dot_scores = torch.matmul(chunk_embs, answer_emb)
 
    # Manhattan similarity (1 / (1 + L1 distance))
    manhattan_dist = torch.sum(torch.abs(chunk_embs - answer_emb), dim=1)
    manhattan_sim = 1 / (1 + manhattan_dist)
 
    return {
        "max_cosine": float(cosine_scores.max()),
        "avg_cosine": float(cosine_scores.mean()),
        "max_euclidean": float(euclidean_sim.max()),
        "avg_euclidean": float(euclidean_sim.mean()),
        "max_dot": float(dot_scores.max()),
        "avg_dot": float(dot_scores.mean()),
        "max_manhattan": float(manhattan_sim.max()),
        "avg_manhattan": float(manhattan_sim.mean()),
        "retrieved_chunks": len(retrieved_chunks)
    }