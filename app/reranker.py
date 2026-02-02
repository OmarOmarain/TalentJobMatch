from sentence_transformers import CrossEncoder

# Initialize Cross Encoder
# ms-marco-MiniLM-L-6-v2 is a good balance of speed and accuracy
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_candidates(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """
    Reranks a list of candidates based on their relevance to the query.
    
    args:
        query: The Job Description.
        candidates: List of candidate dicts (must have 'content').
        top_k: Number of results to return.
    
    returns:
        Top_k candidates with updated 'rerank_score'.
    """
    if not candidates:
        return []

    # Prepare pairs for cross-encoder
    pairs = [[query, c['content']] for c in candidates]
    
    # Predict scores
    scores = model.predict(pairs)
    
    # Attach scores
    for i, candidate in enumerate(candidates):
        candidate['rerank_score'] = float(scores[i])
        
    # Sort by rerank score descending
    sorted_candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
    
    return sorted_candidates[:top_k]
