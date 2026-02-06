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
        return JobDescriptionRequest(description=job) 
    
    raise ValueError(f"Unsupported job input type: {type(job)}")


def _parse_skills(skills_input) -> List[str]:
    """
    Parse skills from various formats to list of strings
    Handles: string, list, or None
    """
    if not skills_input:
        return []
    
    if isinstance(skills_input, list):
        # إذا كانت بالفعل list، تأكد أن كل عنصر string
        return [str(skill).strip() for skill in skills_input]
    
    if isinstance(skills_input, str):
        # إذا كانت string، قسمها على الفواصل
        # التعامل مع تنسيقات مختلفة
        skills_str = skills_input.strip()
        
        # إزالة الأقواس إذا كانت موجودة
        if skills_str.startswith('[') and skills_str.endswith(']'):
            skills_str = skills_str[1:-1]
        
        # تجهيز الفواصل المختلفة
        delimiters = [',', ';', '|', '/', '•', '-', '\n']
        
        for delimiter in delimiters:
            if delimiter in skills_str:
                skills = [skill.strip() for skill in skills_str.split(delimiter) if skill.strip()]
                if skills:
                    return skills
        
        # إذا لم تكن هناك فواصل، رجعها كقائمة من عنصر واحد
        return [skills_str] if skills_str else []
    
    # إذا كان نوع آخر، حوله إلى string أولاً
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
    
    # 1️⃣ Run search pipeline
    search_results = combined_search_pipeline(normalized_job, k=15)
    
    # 2️⃣ Convert to CandidateCard
    candidates = []
    
    for idx, res in enumerate(search_results):
        meta = res.get("metadata", {}) or {}
        
        # DEBUG
        print("\n--- DEBUG SEARCH METADATA ---")
        print(f"Raw meta: {meta}")
        
        # Parse skills safely
        raw_skills = meta.get("skills") or meta.get("top_skills") or []
        skills = _parse_skills(raw_skills)
        
        print(f"Raw skills: {raw_skills}")
        print(f"Parsed skills: {skills}")
        print("--- END DEBUG ---\n")
        
        # Ensure required fields exist with defaults
        candidate_id = meta.get("candidate_id", idx)
        name = meta.get("name", "Unknown")
        years_experience = meta.get("years_experience", 0)
        
        # Parse years_experience to float if needed
        if isinstance(years_experience, str):
            try:
                # استخرج الأرقام من string
                import re
                numbers = re.findall(r'\d+\.?\d*', years_experience)
                years_experience = float(numbers[0]) if numbers else 0.0
            except:
                years_experience = 0.0
        elif not isinstance(years_experience, (int, float)):
            years_experience = 0.0
        
        candidates.append(
            CandidateCard(
                candidate_id=str(candidate_id),
                name=str(name),
                avatar_url=meta.get("avatar_url"),
                current_title=str(meta.get("title", "Unknown")),
                company=str(meta.get("company", "")),
                years_experience=float(years_experience),
                seniority_level=str(meta.get("seniority", "Unknown")),
                location=str(meta.get("location", "")),
                score=float(res.get("score", 0.0)),
                skills_match=skills,  # ✅ الآن تأتي كـ List[str]
                ai_reasoning_short="",
                content=str(res.get("content", "")),
            )
        )
    
    return candidates


# دالة للاختبار
def test_skill_parsing():
    """Test skill parsing function"""
    test_cases = [
        "PHP, Vue.js, Laravel, JavaScript",
        "Python;Django;Flask",
        "React|TypeScript|Node.js",
        "Java Spring Boot",
        ["Python", "ML", "AI"],
        None,
        "",
        "['Python', 'Django', 'React']",
    ]
    
    for test in test_cases:
        result = _parse_skills(test)
        print(f"Input: {test!r}")
        print(f"Output: {result}")
        print(f"Type: {type(result)}")
        print("-" * 40)


if __name__ == "__main__":
    test_skill_parsing()