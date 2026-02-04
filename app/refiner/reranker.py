from typing import List
from dataclasses import dataclass
from sentence_transformers import CrossEncoder
import numpy as np


# ------------------ Mock Models ------------------

@dataclass
class SkillChip:
    name: str


@dataclass
class CandidateCard:
    candidate_id: str
    name: str
    avatar_url: str | None
    current_title: str
    company: str
    years_experience: int
    seniority_level: str
    location: str
    score: float
    skills_match: List[SkillChip]
    ai_reasoning_short: str


# ------------------ Core Function ------------------

model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
cross_encoder = CrossEncoder(model_name)  


def rerank_candidates(job_description: str, candidates: List[CandidateCard]) -> List[CandidateCard]:

    if not candidates:
        return []

    candidate_texts = [
        (
            job_description,
            " ".join(c.skills_match)  # بدل s.name
        )
        for c in candidates
    ]

    scores = cross_encoder.predict(candidate_texts)

    for candidate, score in zip(candidates, scores):
        score_norm = 1 / (1 + np.exp(-score))
        candidate.score = float(score_norm)  # بدل match_score
        candidate.ai_reasoning_short = f"CrossEncoder Score: {candidate.score:.4f}"

    ranked_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)

    return ranked_candidates

    if not candidates:
        return []

    candidate_texts = [
        (
            job_description,
            " ".join(c.skills_match)

        )
        for c in candidates
    ]

    scores = cross_encoder.predict(candidate_texts)

    for candidate, score in zip(candidates, scores):
        score_norm = 1 / (1 + np.exp(-score))
        candidate.score = float(score_norm)
        candidate.ai_reasoning_short = f"CrossEncoder Score: {candidate.score:.4f}"

    ranked_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)

    return ranked_candidates



# ------------------ Local Test ------------------

if __name__ == "__main__":

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
            score=0.0,
            skills_match=[SkillChip(name="React"), SkillChip(name="Vue")],
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
            score=0.0,
            skills_match=[SkillChip(name="Vue")],
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
            score=0.0,
            skills_match=[SkillChip(name="Angular")],
            ai_reasoning_short=""
        ),
    ]

    jd = "Looking for a Frontend Developer skilled in React and Vue"

    ranked = rerank_candidates(jd, candidates)

    print("\n--- Reranked Candidates ---")
    for i, c in enumerate(ranked, 1):
        print(f"{i}. {c.name}")
        print(f"   Score: {c.score:.4f}")
        print(f"   Skills: {[s.name for s in c.skills_match]}")
        print()
