from typing import List
from app.models import CandidateCard
from cross_encoder import CrossEncoder
import numpy as np

def rerank_candidates(job_description: str, candidates: List[CandidateCard]) -> List[CandidateCard]:
    
    if not candidates:
        return []

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder = CrossEncoder(model_name)

    candidate_texts = [
        (
            job_description,
            " ".join([skill.name for skill in c.matching_skills])  
        )
        for c in candidates
    ]

    scores = cross_encoder.predict(candidate_texts)  

    for candidate, score in zip(candidates, scores):

        score_norm = 1 / (1 + np.exp(-score))  # sigmoid
        candidate.match_score = float(score_norm)
        candidate.ai_reasoning_short = f"Score computed by Cross-Encoder: {candidate.match_score}"

    ranked_candidates = sorted(candidates, key=lambda c: c.match_score, reverse=True)

    return ranked_candidates


# ------------------ اختبار سريع ------------------
# if __name__ == "__main__":
    from app.models import CandidateCard, SkillChip

    # مثال Top-K candidates من retrieval
    candidates = [
        CandidateCard(
            candidate_id="1",
            name="Alice",
            avatar_url=None,
            current_title="Frontend Developer",
            company="TechCorp",
            years_experience=3,
            seniority_level="Mid",
            location="Cairo",
            match_score=0.0,  # سيتم استبداله بالـ Cross-Encoder
            matching_skills=[SkillChip(name="React"), SkillChip(name="Vue")],
            ai_reasoning_short=""
        ),
        CandidateCard(
            candidate_id="2",
            name="Bob",
            avatar_url=None,
            current_title="Frontend Engineer",
            company="Innovate",
            years_experience=5,
            seniority_level="Senior",
            location="Cairo",
            match_score=0.0,
            matching_skills=[SkillChip(name="Vue")],
            ai_reasoning_short=""
        ),
        CandidateCard(
            candidate_id="3",
            name="Charlie",
            avatar_url=None,
            current_title="Junior Frontend Developer",
            company="StartUpX",
            years_experience=1,
            seniority_level="Junior",
            location="Cairo",
            match_score=0.0,
            matching_skills=[SkillChip(name="React")],
            ai_reasoning_short=""
        ),
    ]

    jd = "Looking for a Frontend Developer skilled in React and Vue"

    ranked_candidates = rerank_candidates(jd, candidates)

    print("\n--- Reranked Candidates ---")
    for c in ranked_candidates:
        print(f"{c.name}: score={c.match_score}, reasoning={c.ai_reasoning_short}")
