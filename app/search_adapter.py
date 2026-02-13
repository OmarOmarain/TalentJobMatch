from typing import List, Union, Any, Dict
import re
import os
from app.models import CandidateCard, JobDescription, JobDescriptionRequest
from app.search import combined_search_pipeline


def _normalize_job_input(job):
    if isinstance(job, JobDescription):
        return job
    if isinstance(job, JobDescriptionRequest):
        return job  
    if isinstance(job, str):
        return JobDescriptionRequest(description=job) 
    raise ValueError(f"Unsupported job input type: {type(job)}")

def _parse_skills(skills_input: Any) -> List[str]:
    if not skills_input: return []
    if isinstance(skills_input, list):
        return [str(item).strip() for item in skills_input if item]
    if isinstance(skills_input, str):
        skills_str = skills_input.strip()
        if not skills_str: return []
        skills_str = re.sub(r'[\[\]{}()]', '', skills_str)
        delimiters = [',', ';', '|', '/', '•', '-', '\n']
        for d in delimiters:
            if d in skills_str:
                return [p.strip() for p in skills_str.split(d) if p.strip()]
    return [str(skills_input).strip()]

def search_pipeline_to_candidates(
    job: Union[str, JobDescription, JobDescriptionRequest]
) -> List[CandidateCard]:
    """
    1. Normalize job input
    2. Run the combined search pipeline
    3. Convert results to CandidateCard
    4. Deduplicate candidates by candidate_id
    """
    normalized_job = _normalize_job_input(job)
    
    search_results = combined_search_pipeline(normalized_job, k=20)
    
    unique_candidates_map: Dict[str, CandidateCard] = {}
    
    for idx, res in enumerate(search_results):
        meta = res.get("metadata", {}) or {}
        candidate_id = str(meta.get("candidate_id", f"unknown_{idx}"))
        
        if candidate_id in unique_candidates_map:
            continue

        raw_skills = meta.get('top_skills') or meta.get('skills') or []
        skills = _parse_skills(raw_skills)
        
        name = meta.get("name", f"Candidate_{idx}")
        if (name.lower() == "unknown candidate" or name.startswith("Candidate_")) and "source" in meta:
            name = os.path.splitext(meta["source"])[0].replace("_", " ").title()

        raw_exp = meta.get("years_of_experience", 0)
        try:
            if isinstance(raw_exp, str):
                numbers = re.findall(r'\d+\.?\d*', raw_exp)
                years_experience = float(numbers[0]) if numbers else 0.0
            else:
                years_experience = float(raw_exp)
        except:
            years_experience = 0.0

        try:
            candidate = CandidateCard(
                candidate_id=candidate_id,
                name=name,
                avatar_url=meta.get("avatar_url"),
                current_title=str(meta.get("job_title", meta.get("title", "Unknown"))),
                company=str(meta.get("company", "")),
                years_experience=years_experience,
                seniority_level=str(meta.get("seniority_level", "Unknown")),
                location=str(meta.get("location", "")),
                score=float(res.get("score", 0.0)),
                skills_match=skills,
                ai_reasoning_short=""
            )
            
            unique_candidates_map[candidate_id] = candidate
            print(f" Added Unique Candidate: {name} (Score: {candidate.score:.4f})")
            
        except Exception as e:
            print(f" Error mapping candidate {name}: {e}")

    final_candidates = list(unique_candidates_map.values())
    final_candidates.sort(key=lambda x: x.score, reverse=True)

    print(f"\n{'='*60}")
    print(f" Final Unique Candidates: {len(final_candidates)} (Filtered from {len(search_results)})")
    print(f"{'='*60}\n")
    
    return final_candidates