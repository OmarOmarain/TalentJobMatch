import os
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from app.core import get_llm

from app.models import (
    CandidateCard,
    CandidateDeepDive,
    ExplainabilityAnalysis,
    IdentifiedSkill,
    RequirementEvidence,
    BatchEvaluationResponse,
    CandidateEvaluation
)

llm = get_llm(temperature=0.0)

def generate_and_evaluate_batch(
    description: str,
    job_requirements: List[str],
    candidates: List[CandidateCard]
) -> List[CandidateDeepDive]:
    
    if not candidates:
        return []

    results: List[CandidateDeepDive] = []
    
    candidates_data_block = ""
    for c in candidates:
        skills_str = ", ".join(c.skills_match) if isinstance(c.skills_match, list) else str(c.skills_match)
        candidates_data_block += f"CANDIDATE_ID: {c.candidate_id}\nName: {c.name}\nSkills: {skills_str}\n---\n"

    prompt = ChatPromptTemplate.from_template("""
### ROLE
You are an expert Senior HR Technical Auditor. Your mission is to conduct a strict, evidence-based audit. 

### THE GOLDEN RULE: ZERO EXTERNAL KNOWLEDGE
- **GROUNDEDNESS**: You MUST evaluate candidates based ONLY on the text provided in the "CANDIDATES TO AUDIT" section. 
- **NO ASSUMPTIONS**: If a skill, tool, or certification is not explicitly mentioned in the candidate's text, you MUST treat it as non-existent. 
- **NO INFERENCE**: Do not infer that a candidate knows a tool just because they have a certain job title. Evidence must be present in the text.

### AUDIT GUIDELINES
1. **DEEP SCAN**: Meticulously check 'Skills', 'Additional Info', and 'Projects' sections. These are part of your Base Knowledge.
2. **LANGUAGE**: Write the 'summary' in professional ENGLISH only.
3. **ID INTEGRITY**: Return the exact `candidate_id` without any changes.

### EVALUATION METRICS
- **Relevancy Score (0.0 - 1.0)**: Alignment between the candidate's documented skills and the JD.
- **Faithfulness Score (0.0 - 1.0)**: How strictly you followed the Golden Rule (1.0 = 100% based on provided text, 0.0 = contains hallucinations).

### INPUT DATA
---
**JOB DESCRIPTION:**
{description}
---
**CANDIDATES TO AUDIT (YOUR ONLY SOURCE OF TRUTH):**
{candidates_block}
---

### OUTPUT FORMAT
Return valid JSON only:
{{
  "evaluations": [
    {{
      "candidate_id": "string",
      "summary": "Evidence-based analysis: [Strengths based on text] vs [Missing requirements].",
      "faithfulness_score": float,
      "relevancy_score": float
    }}
  ]
}}
""")

    try:
        structured_llm = llm.with_structured_output(BatchEvaluationResponse)
        formatted_prompt = prompt.format(description=description, candidates_block=candidates_data_block)
        
        batch_output = structured_llm.invoke(formatted_prompt)
        
        eval_lookup = {str(e.candidate_id).strip(): e for e in batch_output.evaluations}

        for candidate in candidates:
            c_id = str(candidate.candidate_id).strip()
            evaluation = eval_lookup.get(c_id)
            
            if not evaluation:
                for key, val in eval_lookup.items():
                    if key in c_id or c_id in key:
                        evaluation = val
                        break

            if not evaluation:
                continue

            identified_skills = [
                IdentifiedSkill(skill=s, evidence="Found in resume") 
                for s in candidate.skills_match
            ]

            candidate_skills_lower = [s.lower() for s in candidate.skills_match]
            requirements_comparison = [
                RequirementEvidence(
                    requirement=req,
                    candidate_evidence="Verified in skills list" if any(req.lower() in s for s in candidate_skills_lower) else "Not explicitly mentioned",
                    status="met" if any(req.lower() in s for s in candidate_skills_lower) else "not_met"
                ) for req in job_requirements
            ]

            results.append(CandidateDeepDive(
                candidate_id=candidate.candidate_id,
                explainability=ExplainabilityAnalysis(
                    why_match_summary=evaluation.summary,
                    identified_skills=identified_skills,
                    requirements_comparison=requirements_comparison
                ),
                faithfulness_score=evaluation.faithfulness_score,
                relevancy_score=evaluation.relevancy_score,
                is_trustworthy=(evaluation.faithfulness_score >= 0.75 and evaluation.relevancy_score >= 0.70)
            ))

    except Exception as e:
        print(f"Audit process failed: {e}")
        
    return results