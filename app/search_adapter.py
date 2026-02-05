from typing import List
from app.models import CandidateCard

def search_results_to_candidates(results: List[dict]) -> List[CandidateCard]:
    candidates = []

    for idx, res in enumerate(results):
        meta = res.get("metadata", {})

        # ✅ DEBUG CHECK
        print("\n--- DEBUG SEARCH METADATA ---")
        print(meta)

        # إذا ما في skills، حاول نجيبها من top_skills
        skills = meta.get("skills")
        if skills is None:
            skills = meta.get("top_skills", [])

        # بعض الحقول الأساسية، تعويض إذا غير موجود
        candidate_id = meta.get("candidate_id", idx)
        name = meta.get("name", "Unknown")
        years_experience = meta.get("years_experience", 0)

        candidates.append(
            CandidateCard(
                candidate_id=str(candidate_id),
                name=name,
                avatar_url=meta.get("avatar_url"),
                current_title=meta.get("title", "Unknown"),
                company=meta.get("company", ""),
                years_experience=years_experience,
                seniority_level=meta.get("seniority", "Unknown"),
                location=meta.get("location", ""),
                score=float(res.get("score", 0.0)),
                skills_match=skills,
                ai_reasoning_short="",
                content=res.get("content", "")
            )
        )

    return candidates
