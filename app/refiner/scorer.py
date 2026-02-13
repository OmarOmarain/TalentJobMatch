import logging
from typing import List, Dict

import numpy as np
from sentence_transformers import CrossEncoder

from app.models import CandidateCard, CandidateDeepDive


logger = logging.getLogger("Scorer")


# تحميل موديل CrossEncoder مرة واحدة
try:
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder = CrossEncoder(model_name)
    logger.info(f"CrossEncoder '{model_name}' loaded successfully.")

except Exception as e:
    logger.error(f"CrossEncoder failed to load: {e}")
    cross_encoder = None


def process_and_score_candidates(
        jd_text: str,
        candidates: List[CandidateCard],
        evaluations: List[CandidateDeepDive] = None
) -> List[Dict]:

    if not candidates:
        logger.warning("No candidates provided for scoring.")
        return []

    # ---------- Semantic Matching ----------

    candidate_texts = [
        (jd_text, f"{c.current_title} " + ", ".join(c.skills_match))
        for c in candidates
    ]

    if cross_encoder:
        raw_rerank_scores = cross_encoder.predict(candidate_texts)
    else:
        raw_rerank_scores = [0.0] * len(candidates)

    # ---------- Map evaluations ----------
    eval_map = {
        str(e.candidate_id).strip(): e
        for e in evaluations
    } if evaluations else {}

    final_results = []

    for c, raw_score in zip(candidates, raw_rerank_scores):

        curr_id = str(c.candidate_id).strip()

        # تحويل إلى قيمة بين 0 و 1
        semantic_match = float(1 / (1 + np.exp(-raw_score)))

        ev = eval_map.get(curr_id)

        relevancy = float(ev.relevancy_score) if ev else semantic_match
        faithfulness = float(ev.faithfulness_score) if ev else 1.0

        # ---------- Final Weighted Score ----------
        combined_base_score = (0.3 * semantic_match) + (0.7 * relevancy)

        percentage_score = round(float(combined_base_score * 100), 1)

        final_percentage = float(np.clip(percentage_score, 1.0, 99.9))

        # ---------- Reasoning ----------
        if ev and ev.explainability and ev.explainability.why_match_summary:
            ai_reasoning = ev.explainability.why_match_summary
        else:
            skills_preview = ", ".join(c.skills_match[:3]) if c.skills_match else "profile keywords"
            ai_reasoning = f"Profile shows strong semantic alignment with focus on {skills_preview}."

        # ---------- Requirements Comparison ----------
        requirements = ev.requirements_comparison if ev else []

        final_results.append({
            "candidate_id": curr_id,
            "name": str(c.name),
            "score": final_percentage,
            "skills_match": c.skills_match,
            "reasoning": ai_reasoning,
            "faithfulness_score": faithfulness,
        })

    return sorted(final_results, key=lambda x: x["score"], reverse=True)
