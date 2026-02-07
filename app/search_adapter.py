"""
Integrated Search + Candidate Mapping
This module uses combined_search_pipeline to fetch documents from the vector store,
then converts the results to CandidateCard objects using search_results_to_candidates.
"""

from typing import List, Union, Any
import re
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
    """
    Parse skills from various formats to list of strings
    Handles: string, list, or None
    """
    if not skills_input:
        return []
    
    if isinstance(skills_input, list):
        result = []
        for item in skills_input:
            if item is None:
                continue
            item_str = str(item).strip()
            if item_str:
                result.append(item_str)
        return result
    
    if isinstance(skills_input, str):
        skills_str = skills_input.strip()
        
        if not skills_str:
            return []
        
        print(f"🔍 DEBUG _parse_skills input: {skills_str!r}")
        
        if skills_str.startswith('[') and skills_str.endswith(']'):
            skills_str = skills_str[1:-1].strip()
        elif skills_str.startswith('(') and skills_str.endswith(')'):
            skills_str = skills_str[1:-1].strip()
        elif skills_str.startswith('{') and skills_str.endswith('}'):
            skills_str = skills_str[1:-1].strip()
        
        if not skills_str:
            return []
        
        delimiters = [',', ';', '|', '/', '•', '-', '\n', '\\n']
        
        for delimiter in delimiters:
            if delimiter in skills_str:
                parts = [part.strip() for part in skills_str.split(delimiter) if part.strip()]
                if parts:
                    print(f" Split on '{delimiter}': {parts}")
                    return parts
        
        if ' and ' in skills_str.lower() or ' or ' in skills_str.lower():
            parts = re.split(r'\s+(?:and|or)\s+', skills_str, flags=re.IGNORECASE)
            parts = [p.strip() for p in parts if p.strip()]
            if parts:
                print(f"✅ Split on 'and/or': {parts}")
                return parts
        
        print(f"✅ Single skill: [{skills_str}]")
        return [skills_str]
    
    return [str(skills_input).strip()]


def search_pipeline_to_candidates(
    job: Union[str, JobDescription, JobDescriptionRequest]
) -> List[CandidateCard]:
    """
    1. Normalize job input
    2. Run the combined search pipeline
    3. Convert results to CandidateCard
    """
    
    normalized_job = _normalize_job_input(job)
    
    search_results = combined_search_pipeline(normalized_job, k=15)
    
    candidates = []
    
    for idx, res in enumerate(search_results):
        meta = res.get("metadata", {}) or {}
        
        print(f"\n{'='*60}")
        print(f" Processing candidate #{idx + 1}")
        print(f"{'='*60}")
        
        raw_skills = None
        skill_field_used = None
        
        possible_skill_fields = [
            'skills', 'top_skills', 'skill', 'skills_match',
            'technologies', 'technology', 'tools', 'expertise',
            'competencies', 'qualifications', 'proficiencies'
        ]
        
        for field in possible_skill_fields:
            if field in meta:
                raw_skills = meta[field]
                skill_field_used = field
                print(f" Found skills in field '{field}': {raw_skills!r}")
                print(f" Type of raw_skills: {type(raw_skills)}")
                break
        
        if raw_skills is None:
            raw_skills = []
            print(" No skills field found in metadata")
        else:
            print(f"✅ Using skills from field: {skill_field_used}")
        
        skills = _parse_skills(raw_skills)
        print(f"✅ Parsed skills ({len(skills)} items): {skills}")
        print(f"✅ Type of parsed skills: {type(skills)}")
        
        candidate_id = meta.get("candidate_id", idx)
        name = meta.get("name", f"Candidate_{idx}")
        
        years_experience = meta.get("years_experience", 0)
        if isinstance(years_experience, str):
            try:
                numbers = re.findall(r'\d+\.?\d*', years_experience)
                years_experience = float(numbers[0]) if numbers else 0.0
            except:
                years_experience = 0.0
        elif not isinstance(years_experience, (int, float)):
            years_experience = 0.0
        
        if not isinstance(skills, list):
            print(f" CRITICAL: skills is not a list! Type: {type(skills)}")
            print(f" Value: {skills!r}")
            skills = []  
        
        try:
            candidate = CandidateCard(
                candidate_id=str(candidate_id),
                name=str(name),
                avatar_url=meta.get("avatar_url"),
                current_title=str(meta.get("title", "Unknown")),
                company=str(meta.get("company", "")),
                years_experience=float(years_experience),
                seniority_level=str(meta.get("seniority", "Unknown")),
                location=str(meta.get("location", "")),
                score=float(res.get("score", 0.0)),
                skills_match=skills,  
                ai_reasoning_short="",
                content=str(res.get("content", "")),
            )
            
            candidates.append(candidate)
            print(f" Successfully created CandidateCard for: {candidate.name}")
            
        except Exception as e:
            print(f" ERROR creating CandidateCard: {e}")
            print(f" skills value that caused error: {skills!r}")
            print(f" skills type: {type(skills)}")
            
            try:
                candidate = CandidateCard(
                    candidate_id=str(candidate_id),
                    name=str(name),
                    avatar_url=meta.get("avatar_url"),
                    current_title=str(meta.get("title", "Unknown")),
                    company=str(meta.get("company", "")),
                    years_experience=float(years_experience),
                    seniority_level=str(meta.get("seniority", "Unknown")),
                    location=str(meta.get("location", "")),
                    score=float(res.get("score", 0.0)),
                    skills_match=[], 
                    ai_reasoning_short="",
                    content=str(res.get("content", "")),
                )
                candidates.append(candidate)
                print(f" Created CandidateCard with empty skills list")
            except:
                print(f" Failed to create CandidateCard even with empty skills")
    
    print(f"\n{'='*60}")
    print(f" Total candidates created: {len(candidates)}")
    print(f"{'='*60}\n")
    
    return candidates


if __name__ == "__main__":
    print("🧪 Testing skill parsing with problematic input...")
    
    test_metadata = {
        "skills": "WordPress, Elementor, WP, Element, HTML, CSS, PHP, SQL",
        "name": "Test Candidate",
        "title": "Web Developer",
        "candidate_id": 1
    }
    
    mock_result = {
        "metadata": test_metadata,
        "score": 0.85,
        "content": "Test content"
    }
    
    raw = test_metadata["skills"]
    print(f"\nTest input: {raw!r}")
    print(f"Type: {type(raw)}")
    
    parsed = _parse_skills(raw)
    print(f"Parsed: {parsed}")
    print(f"Parsed type: {type(parsed)}")
    print(f"Is list? {isinstance(parsed, list)}")
    
    print(f"\n{'='*60}")
    print("Testing CandidateCard creation...")
    
    try:
        candidate = CandidateCard(
            candidate_id="1",
            name="Test",
            current_title="Developer",
            company="Test Co",
            years_experience=5.0,
            seniority_level="Mid",
            location="Remote",
            score=0.85,
            skills_match=parsed,  
            ai_reasoning_short="",
            content="Test"
        )
        print(f" Success! Candidate created with skills: {candidate.skills_match}")
    except Exception as e:
        print(f" Error: {e}")