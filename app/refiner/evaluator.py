import asyncio
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

BATCH_SIZE = 3
MAX_PARALLEL = 3


def chunk_list(data, size):
    for i in range(0, len(data), size):
        yield data[i:i + size]


async def _evaluate_chunk(
    description: str,
    job_requirements: List[str],
    candidates: List[CandidateCard],
    semaphore: asyncio.Semaphore
) -> List[CandidateDeepDive]:

    async with semaphore:

        results: List[CandidateDeepDive] = []

        candidates_block = ""

        for c in candidates:
            skills_str = ", ".join(c.skills_match)
            candidates_block += (
                f"CANDIDATE_ID: {c.candidate_id}\n"
                f"Name: {c.name}\n"
                f"Skills: {skills_str}\n---\n"
            )

        prompt = ChatPromptTemplate.from_template("""
ROLE: HR Auditor

JOB DESCRIPTION:
{description}

CANDIDATES:
{candidates_block}

Return JSON:
{{
  "evaluations":[
    {{
      "candidate_id":"string",
      "summary":"string",
      "faithfulness_score":float,
      "relevancy_score":float
    }}
  ]
}}
""")

        structured_llm = llm.with_structured_output(BatchEvaluationResponse)

        formatted = prompt.format(
            description=description,
            candidates_block=candidates_block
        )

        batch_output = await structured_llm.ainvoke(formatted)

        eval_lookup = {
            str(e.candidate_id).strip(): e
            for e in batch_output.evaluations
        }

        for candidate in candidates:

            evaluation = eval_lookup.get(str(candidate.candidate_id).strip())
            if not evaluation:
                continue

            identified_skills = [
                IdentifiedSkill(skill=s, evidence="Found in resume")
                for s in candidate.skills_match
            ]

            candidate_skills_lower = [s.lower() for s in candidate.skills_match]

            requirements_comparison = []

            for req in job_requirements:

                is_met = any(req.lower() in s for s in candidate_skills_lower)

                requirements_comparison.append(
                    RequirementEvidence(
                        requirement=req,
                        candidate_evidence="Verified in skills list" if is_met else "Not mentioned",
                        status="met" if is_met else "not_met"
                    )
                )

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

        return results


async def generate_and_evaluate_batch(
    description: str,
    job_requirements: List[str],
    candidates: List[CandidateCard]
) -> List[CandidateDeepDive]:

    if not candidates:
        return []

    semaphore = asyncio.Semaphore(MAX_PARALLEL)

    tasks = []

    for chunk in chunk_list(candidates, BATCH_SIZE):
        tasks.append(
            _evaluate_chunk(description, job_requirements, chunk, semaphore)
        )

    results = await asyncio.gather(*tasks)

    # flatten list
    return [item for sublist in results for item in sublist]
