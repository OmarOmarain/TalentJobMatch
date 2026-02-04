from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class JobDescriptionRequest(BaseModel):
    """Lightweight request model for API - only requires description"""
    description: str = Field(min_length=20)


class JobDescription(BaseModel):
    """Internal structured job description model"""
    title: str = Field(default="Unknown", description="Job title")
    description: str = Field(..., description="Full job description text")
    required_skills: List[str] = Field(default_factory=list, description="List of required skills")
    seniority_level: Optional[str] = Field(
        default=None,
        pattern="^(junior|mid|senior|lead)$",
        description="Experience level: junior, mid, senior, lead"
    )
    department: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Department or team"
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
    required_skills: List[str] = Field(default_factory=list, description="List of required skills")
    seniority_level: Optional[str] = Field(
        default=None,
        pattern="^(junior|mid|senior|lead)$",
        description="Experience level: junior, mid, senior, lead"
    )
    years_of_experience: Optional[int] = None
    department: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Department or team"
    )


   
class MatchResult(BaseModel):
    candidate_id: str
    name: str
    score: float = Field(..., description="Overall match score (0-1)")
    skills_match: List[str] = Field(default_factory=list, description="Skills found in both JD and Candidate")
    reasoning: str = Field(..., description="AI explanation for the match")
    faithfulness_score: float = Field(..., description="Faithfulness score of the explanation")

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
class CandidateMetadata(BaseModel):
    summary: str = Field(..., description="Brief 2-3 sentence professional summary")
    top_skills: List[str] = Field(default_factory=list, description="Top 5-10 technical skills found in resume")


class MatchResult(BaseModel):
    candidate_id: str
    name: str
    score: float = Field(..., ge=0, le=1, description="Match score (0-1)")
    skills_match: List[str] = Field(default_factory=list, description="Skills matched between JD and candidate")
    reasoning: Optional[str] = Field(None, description="AI generated reasoning / short justification")
    faithfulness_score: Optional[float] = Field(None, ge=0, le=1, description="Faithfulness of reasoning to CV/evidence")


class MatchResponse(BaseModel):
    total_candidates: int
    top_matches: List[MatchResult]
