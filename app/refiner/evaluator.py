from typing import List
from langchain_core.prompts import ChatPromptTemplate

from app.core import get_llm
from app.models import (
    CandidateCard,
    CandidateDeepDive,
    ExplainabilityAnalysis,
    IdentifiedSkill,
    RequirementEvidence,
    BatchEvaluationResponse
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

    # ---------- Prepare Candidates Block ----------
    candidates_data_block = ""

    for c in candidates:
        skills_str = ", ".join(c.skills_match) if isinstance(c.skills_match, list) else str(c.skills_match)

        candidates_data_block += (
            f"CANDIDATE_ID: {c.candidate_id}\n"
            f"Name: {c.name}\n"
            f"Skills: {skills_str}\n---\n"
        )

    # ---------- Prompt ----------
    prompt = ChatPromptTemplate.from_template("""
### ROLE
You are an expert Senior HR Technical Auditor. Your mission is to conduct a strict, evidence-based audit. 

### THE GOLDEN RULE: ZERO EXTERNAL KNOWLEDGE
- GROUNDEDNESS: You MUST evaluate candidates based ONLY on the text provided in the CANDIDATES TO AUDIT section.
- NO ASSUMPTIONS: If a skill, tool, or certification is not explicitly mentioned, treat it as non-existent.
- NO INFERENCE: Do not infer knowledge from job titles.

### AUDIT GUIDELINES
1. Perform deep scan across skills and profile content.
2. Write summary in professional English.
3. Preserve candidate_id EXACTLY.

### EVALUATION METRICS
- Relevancy Score (0.0 - 1.0)
- Faithfulness Score (0.0 - 1.0)

### INPUT DATA
---
JOB DESCRIPTION:
{description}
---

CANDIDATES TO AUDIT:
{candidates_block}
---

### OUTPUT FORMAT
Return valid JSON only:
{{
  "evaluations": [
    {{
      "candidate_id": "string",
      "summary": "Evidence-based analysis: [Strengths] vs [Missing requirements]",
      "faithfulness_score": float,
      "relevancy_score": float
    }}
  ]
}}
""")


    try:

        structured_llm = llm.with_structured_output(BatchEvaluationResponse)

        formatted_prompt = prompt.format(
            description=description,
            candidates_block=candidates_data_block
        )

        batch_output = structured_llm.invoke(formatted_prompt)

        eval_lookup = {
            str(e.candidate_id).strip(): e
            for e in batch_output.evaluations
        }

        # ---------- Build Deep Dive ----------
        for candidate in candidates:

            c_id = str(candidate.candidate_id).strip()
            evaluation = eval_lookup.get(c_id)

            if not evaluation:
                continue

            # ---------- Identified Skills ----------
            identified_skills = [
                IdentifiedSkill(
                    skill=s,
                    evidence="Found in resume"
                )
                for s in candidate.skills_match
            ]

            # ---------- Requirement Comparison ----------
            candidate_skills_lower = [s.lower() for s in candidate.skills_match]

            requirements_comparison = []

            for req in job_requirements:

                is_met = any(req.lower() in s for s in candidate_skills_lower)

                requirements_comparison.append(
                    RequirementEvidence(
                        requirement=req,
                        candidate_evidence="Verified in skills list" if is_met else "Not explicitly mentioned",
                        status="met" if is_met else "not_met"
                    )
                )

            # ---------- Build DeepDive ----------
            results.append(
                CandidateDeepDive(
                    candidate_id=candidate.candidate_id,

                    explainability=ExplainabilityAnalysis(
                        why_match_summary=evaluation.summary,
                        identified_skills=identified_skills
                    ),

                    faithfulness_score=evaluation.faithfulness_score,
                    relevancy_score=evaluation.relevancy_score,

                    is_trustworthy=(
                        evaluation.faithfulness_score >= 0.75
                        and evaluation.relevancy_score >= 0.70
                    ),

                    requirements_comparison=requirements_comparison
                )
            )

    except Exception as e:
        print(f"Audit process failed: {e}")

    return results
