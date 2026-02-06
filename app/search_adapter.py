"""
Integrated Search + Candidate Mapping
This module uses combined_search_pipeline to fetch documents from the vector store,
then converts the results to CandidateCard objects using search_results_to_candidates.
"""

from typing import List, Union
from app.models import CandidateCard, JobDescription, JobDescriptionRequest
from app.search import combined_search_pipeline



def _normalize_job_input(job):

    if isinstance(job, JobDescription):
        return job

    if isinstance(job, JobDescriptionRequest):
        return job

    if isinstance(job, str):
        return JobDescriptionRequest(
            job_description=job  
        )

    raise ValueError(f"Unsupported job input type: {type(job)}")


def search_pipeline_to_candidates(
    job: Union[str, JobDescription, JobDescriptionRequest]
) -> List[CandidateCard]:
    """
    1. Normalize job input
    2. Run the combined search pipeline
    3. Convert results to CandidateCard
    """

    normalized_job = _normalize_job_input(job)

    # 1️⃣ Run search pipeline
    search_results = combined_search_pipeline(normalized_job, k=15)

    # 2️⃣ Convert to CandidateCard
    candidates = []

    for idx, res in enumerate(search_results):
        meta = res.get("metadata", {}) or {}

        # DEBUG
        print("\n--- DEBUG SEARCH METADATA ---")
        print(meta)

        # Use 'skills' if exists, else fallback to 'top_skills'
        skills = meta.get("skills") or meta.get("top_skills") or []

        # Ensure required fields exist with defaults
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
                content=res.get("content", ""),
            )
        )

    return candidates
