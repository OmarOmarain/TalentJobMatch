from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class JobDescriptionRequest(BaseModel):
    job_description: str = Field(
        ...,
        min_length=20,
        description="Full job description entered by HR"
    )
    language: str = Field(
        default="en",
        description="Language of the job description"
    )


# ---------- Atomic Models ----------

class IdentifiedSkill(BaseModel):
    skill: str = Field(..., description="Skill identified in candidate profile")
    evidence: str = Field(..., description="Evidence supporting the skill")


class RequirementEvidence(BaseModel):
    requirement: str = Field(..., description="Job requirement")
    candidate_evidence: str = Field(..., description="Evidence from candidate CV")
    status: Literal["met", "partial", "not_met"]


# ---------- Explainability ----------

class ExplainabilityAnalysis(BaseModel):
    why_match_summary: str = Field(
        ..., description="LLM explanation of why the candidate matches the role"
    )

    identified_skills: List[IdentifiedSkill] = Field(
        default_factory=list,
        description="Detailed skill extraction with evidence"
    )

    requirements_comparison: List[RequirementEvidence] = Field(
        default_factory=list,
        description="Requirement-by-requirement evaluation"
    )


# ---------- Candidate Surface Card (for ranking / UI) ----------

class CandidateCard(BaseModel):
    candidate_id: str

    name: str
    avatar_url: Optional[str] = None

    current_title: str
    company: str

    years_experience: int
    seniority_level: str

    location: Optional[str] = None

    score: float = Field(
        ..., ge=0, le=1,
        description="Overall match score (0-1)"
    )

    skills_match: List[str] = Field(
        default_factory=list,
        description="Skills found in both JD and Candidate"
    )

    ai_reasoning_short: str = Field(
        ..., description="Short AI-generated justification (UI-safe)"
    )


# ---------- Deep Analysis (internal / evaluation layer) ----------

class CandidateDeepDive(BaseModel):
    candidate_id: str

    explainability: ExplainabilityAnalysis

    faithfulness_score: float = Field(
        ..., ge=0, le=1,
        description="How grounded the explanation is in the CV evidence"
    )

    relevancy_score: float = Field(
        ..., ge=0, le=1,
        description="How relevant the explanation is to the job description"
    )

    is_trustworthy: bool = Field(
        ..., description="Passed faithfulness & relevancy thresholds"
    )


# ---------- API Responses ----------

class RankingResponse(BaseModel):
    job_description: str

    total_candidates_scanned: int

    top_candidates: List[CandidateCard]


class CandidateAnalysisResponse(BaseModel):
    candidate: CandidateCard
    deep_dive: CandidateDeepDive
