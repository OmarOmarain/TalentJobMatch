from typing import List
from app.models import CandidateCard


# def search_results_to_candidates(results: List[dict]) -> List[CandidateCard]:

    # candidates = []

    # for idx, res in enumerate(results):

    #     meta = res.get("metadata", {})

    #     candidates.append(
    #         CandidateCard(
    #             candidate_id=str(meta.get("candidate_id", idx)),
    #             name=meta.get("name", "Unknown"),
    #             avatar_url=meta.get("avatar_url"),
    #             current_title=meta.get("title", "Unknown"),
    #             company=meta.get("company", ""),
    #             years_experience=meta.get("years_experience", 0),
    #             seniority_level=meta.get("seniority", "Unknown"),
    #             location=meta.get("location", ""),
    #             score=float(res.get("score", 0.0)),
    #             skills_match=meta.get("skills", []),
    #             ai_reasoning_short="",
    #             content=res.get("content", "")
    #         )
    #     )

    # return candidates


def search_results_to_candidates(results):

    candidates = []

    for idx, res in enumerate(results):

        meta = res.get("metadata", {})

        # ✅ DEBUG CHECK
        print("\n--- DEBUG SEARCH METADATA ---")
        print(meta)

        required_fields = ["skills", "years_experience", "name", "candidate_id"]

        for field in required_fields:
            if field not in meta:
                raise ValueError(f"Missing metadata field: {field}")

        candidates.append(
            CandidateCard(
                candidate_id=str(meta["candidate_id"]),
                name=meta["name"],
                avatar_url=meta.get("avatar_url"),
                current_title=meta.get("title", "Unknown"),
                company=meta.get("company", ""),
                years_experience=meta["years_experience"],
                seniority_level=meta.get("seniority", "Unknown"),
                location=meta.get("location", ""),
                score=float(res.get("score", 0.0)),
                skills_match=meta["skills"],
                ai_reasoning_short="",
                content=res.get("content", "")
            )
        )

    return candidates
