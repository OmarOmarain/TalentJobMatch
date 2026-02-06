import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from app.models import (
    CandidateCard,
    CandidateDeepDive,
    ExplainabilityAnalysis,
    IdentifiedSkill,
    RequirementEvidence
)


# ------------------ LLM ------------------
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.2
)


# ------------------ Core Function ------------------

def generate_explanations(
    job_description: str,
    job_requirements: List[str],
    candidates: List[CandidateCard]
) -> List[CandidateDeepDive]:

    results: List[CandidateDeepDive] = []

    for candidate in candidates:

        # -------- Prompt --------
        prompt = f"""
You are an AI hiring assistant.

STRICT RULES:
- Use ONLY the provided information
- Do NOT assume missing skills
- If evidence does not exist, say so explicitly

JOB DESCRIPTION:
{job_description}

JOB REQUIREMENTS:
{job_requirements}

CANDIDATE DATA:
Name: {candidate.name}
Current Title: {candidate.current_title}
Years of Experience: {candidate.years_experience}
Skills: {candidate.skills_match}

TASK:
Explain briefly why this candidate matches or does not match the role.
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        explanation_text = response.content

        # -------- Identified Skills --------
        identified_skills = [
            IdentifiedSkill(
                skill=skill,
                evidence="Explicitly listed in candidate CV"
            )
            for skill in candidate.skills_match
        ]

        # -------- Requirements Comparison --------
        candidate_skill_names = {
            skill.lower() for skill in candidate.skills_match
        }

        requirements_comparison: List[RequirementEvidence] = []

        for requirement in job_requirements:

            if requirement.lower() in candidate_skill_names:
                status = "met"
                evidence = "Skill explicitly present in CV"
            else:
                status = "not_met"
                evidence = "No evidence found in CV"

            requirements_comparison.append(
                RequirementEvidence(
                    requirement=requirement,
                    candidate_evidence=evidence,
                    status=status
                )
            )

        # -------- Explainability Object --------
        explainability = ExplainabilityAnalysis(
            why_match_summary=explanation_text,
            identified_skills=identified_skills,
            requirements_comparison=requirements_comparison
        )

        # -------- Deep Dive --------
        deep_dive = CandidateDeepDive(
            candidate_id=candidate.candidate_id,
            explainability=explainability,
            relevancy_score=0.0,
            faithfulness_score=0.0,
            is_trustworthy=False
        )

        results.append(deep_dive)

    return results
