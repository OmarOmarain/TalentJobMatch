from typing import List, Dict
import numpy as np
import logging
from sentence_transformers import CrossEncoder
from app.models import CandidateCard, CandidateDeepDive

logger = logging.getLogger("Scorer")

# Load model once globally
try:
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder = CrossEncoder(model_name)
except Exception as e:
    logger.error(f"CrossEncoder failed to load: {e}")
    cross_encoder = None

def process_and_score_candidates(
    jd_text: str, 
    candidates: List[CandidateCard], 
    evaluations: List[CandidateDeepDive] = None
) -> List[Dict]:
    """
    Calculates final scores and reasoning, strictly adhering to Pydantic constraints.
    Score range: 0.0 to 1.0 (as required by MatchResult)
    """
    if not candidates:
        return []

    # 1. Semantic Reranking
    candidate_texts = [(jd_text, f"{c.current_title} " + ", ".join(c.skills_match)) for c in candidates]
    
    if cross_encoder:
        raw_rerank_scores = cross_encoder.predict(candidate_texts)
    else:
        raw_rerank_scores = [0.0] * len(candidates)

    # 2. Evaluation Mapping (ID cleaning to ensure match)
    eval_map = {str(e.candidate_id).strip(): e for e in evaluations} if evaluations else {}

    final_results = []

    for idx, (c, raw_score) in enumerate(zip(candidates, raw_rerank_scores)):
        curr_id = str(c.candidate_id).strip()
        
        # Calculate semantic score via Sigmoid (0 to 1)
        semantic_match = float(1 / (1 + np.exp(-raw_score)))
        
        # Fetch LLM DeepDive
        ev = eval_map.get(curr_id)
        
        # Determine metrics with safe fallbacks
        relevancy = float(ev.relevancy_score) if ev else semantic_match
        faithfulness = float(ev.faithfulness_score) if ev else 1.0
        
        # Final calibrated score: 0.3 Semantic + 0.7 Analyst Relevancy
        combined_score = (0.3 * semantic_match) + (0.7 * relevancy)
        
        # CRITICAL: Keep score between 0.0 and 1.0 for Pydantic MatchResult
        final_score = float(np.clip(combined_score, 0.01, 0.99))

        # 3. Reasoning fallback mechanism to prevent empty strings
        ai_reasoning = ""
        if ev and ev.explainability and ev.explainability.why_match_summary:
            ai_reasoning = ev.explainability.why_match_summary
        else:
            # Automatic generation if Gemini fails or quota is hit
            skills_preview = ", ".join(c.skills_match[:3]) if c.skills_match else "relevant experience"
            ai_reasoning = f"Matched based on profile similarity with focus on {skills_preview}."
        # Building the dict that matches 'MatchResult' Pydantic model
        final_results.append({
            "candidate_id": curr_id,
            "name": str(c.name),
            "score": final_score,
            "skills_match": c.skills_match,
            "reasoning": ai_reasoning,
            "faithfulness_score": faithfulness
        })

    # Sort descending by score
    return sorted(final_results, key=lambda x: x["score"], reverse=True)