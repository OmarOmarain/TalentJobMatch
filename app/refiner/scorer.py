from typing import List
from app.models import CandidateCard


def calculate_match_scores(candidates: List[CandidateCard]) -> List[CandidateCard]:

    if not candidates:
        return []

    for c in candidates:

        base = c.score
        skills = min(len(c.skills_match) / 5, 1)
        experience = min(c.years_experience / 10, 1)

        # -------- Final weighted score --------
        final_score = 0.5 * base + 0.3 * skills + 0.2 * experience

        # -------- Calibration (BOOST readability) --------
        final_score = final_score ** 0.7

        c.score = float(final_score)

        c.ai_reasoning_short += (
            f" | Skills={skills:.2f}, Experience={experience:.2f}"
        )

    return sorted(candidates, key=lambda c: c.score, reverse=True)
