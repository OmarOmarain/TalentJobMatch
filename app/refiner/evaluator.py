import os
from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from app.models import (
    CandidateCard,
    CandidateDeepDive,
    ExplainabilityAnalysis,
    IdentifiedSkill,
    RequirementEvidence
)

load_dotenv()

# Internal schema for structured extraction
class EvaluationSchema(BaseModel):
    summary: str = Field(description="A brief explanation of why the candidate matches or doesn't match the job.")
    faithfulness_score: float = Field(description="Score between 0.0 and 1.0 indicating how well the explanation is grounded in the candidate data.")
    relevancy_score: float = Field(description="Score between 0.0 and 1.0 indicating how relevant the candidate is to the job description.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0
)

def generate_and_evaluate_batch(
    description: str,
    job_requirements: List[str],
    candidates: List[CandidateCard]
) -> List[CandidateDeepDive]:
    
    results: List[CandidateDeepDive] = []
    
    # Binding the LLM to the schema for guaranteed JSON output
    structured_llm = llm.with_structured_output(EvaluationSchema)

    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert HR Analyst. Analyze the following candidate for a specific job role.
    
    JOB DESCRIPTION:
    {description}
    
    CANDIDATE DATA:
    Name: {name}
    Current Title: {title}
    Years of Experience: {exp}
    Skills: {skills}
    
    INSTRUCTIONS:
    1. Evaluate the match summary.
    2. Provide scores for faithfulness and relevancy.
    3. Be objective and strictly base your evaluation on the candidate data provided.
    """)

    for candidate in candidates:
        try:
            # Single optimized LLM call per candidate
            evaluation = structured_llm.invoke(
                prompt_template.format(
                    description=description,
                    name=candidate.name,
                    title=candidate.current_title,
                    exp=candidate.years_experience,
                    skills=", ".join(candidate.skills_match) if isinstance(candidate.skills_match, list) else str(candidate.skills_match)
                )
            )

            # Map the skills to IdentifiedSkill model
            identified_skills = [
                IdentifiedSkill(skill=s, evidence="Found in Candidate Profile") 
                for s in candidate.skills_match
            ]

            # Compare requirements against candidate skills
            candidate_skills_lower = {s.lower() for s in candidate.skills_match}
            requirements_comparison = [
                RequirementEvidence(
                    requirement=req,
                    candidate_evidence="Explicitly mentioned" if req.lower() in candidate_skills_lower else "Not found",
                    status="met" if req.lower() in candidate_skills_lower else "not_met"
                ) for req in job_requirements
            ]

            # Constructing the final DeepDive object
            deep_dive = CandidateDeepDive(
                candidate_id=candidate.candidate_id,
                explainability=ExplainabilityAnalysis(
                    why_match_summary=evaluation.summary,
                    identified_skills=identified_skills,
                    requirements_comparison=requirements_comparison
                ),
                faithfulness_score=evaluation.faithfulness_score,
                relevancy_score=evaluation.relevancy_score,
                is_trustworthy=(evaluation.faithfulness_score >= 0.75 and evaluation.relevancy_score >= 0.70)
            )
            
            results.append(deep_dive)

        except Exception as e:
            print(f"Failed to evaluate candidate {candidate.name}: {e}")
            continue

    return results