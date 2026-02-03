from typing import List
from app.models import CandidateCard


def calculate_match_scores(
    candidates: List[CandidateCard]
) -> List[CandidateCard]:
    """
    Combines semantic score + skills + experience into final 0–100 score.
    """

    if not candidates:
        return []

    for c in candidates:
        base = c.semantic_score                       # 0–1
        skills = min(len(c.matching_skills) / 5, 1)  # normalize
        experience = min(c.years_experience / 10, 1)

        final_score = (
            0.5 * base +
            0.3 * skills +
            0.2 * experience
        )

        c.match_score = int(final_score * 100)

        c.ai_reasoning_short = (
            f"Semantic={base:.2f}, "
            f"Skills={skills:.2f}, "
            f"Experience={experience:.2f}"
        )

    return sorted(
        candidates,
        key=lambda c: c.match_score,
        reverse=True
    )
