from typing import List, Literal, Optional
from pydantic import BaseModel, Field, validator


# ---------- Job Description ----------

class JobDescriptionRequest(BaseModel):
    description: str = Field(min_length=20)


class JobDescription(BaseModel):
    title: str = Field(default="Unknown")
    description: str
    required_skills: List[str] = Field(default_factory=list)

    seniority_level: Optional[str] = Field(
        default=None,
        pattern="^(junior|mid|senior|lead)$"
    )

    department: Optional[str] = Field(default=None, max_length=100)


# ---------- Atomic Models ----------

class IdentifiedSkill(BaseModel):
    skill: str
    evidence: str


class RequirementEvidence(BaseModel):
    requirement: str
    candidate_evidence: str
    status: Literal["met", "partial", "not_met"]


# ---------- Explainability ----------

class ExplainabilityAnalysis(BaseModel):

    why_match_summary: str

    identified_skills: List[IdentifiedSkill] = Field(default_factory=list)

    required_skills: List[str] = Field(default_factory=list)

    seniority_level: Optional[str] = Field(
        default=None,
        pattern="^(junior|mid|senior|lead)$"
    )

    years_of_experience: Optional[int] = None

    department: Optional[str] = Field(default=None, max_length=100)


# ---------- Candidate Surface Card ----------

class CandidateCard(BaseModel):

    candidate_id: str
    name: str
    avatar_url: Optional[str] = None

    current_title: str
    company: str

    years_experience: int
    seniority_level: str

    location: Optional[str] = None

    score: float = Field(..., ge=0, le=1)

    skills_match: List[str] = Field(default_factory=list)

    ai_reasoning_short: str


# ---------- Deep Analysis ----------

class CandidateDeepDive(BaseModel):

    candidate_id: str

    explainability: ExplainabilityAnalysis

    faithfulness_score: float = Field(..., ge=0, le=1)

    relevancy_score: float = Field(..., ge=0, le=1)

    is_trustworthy: bool

    requirements_comparison: List[RequirementEvidence] = Field(default_factory=list)


# ---------- Final Match Result ----------

class MatchResult(BaseModel):

    candidate_id: str
    name: str

    # نسبة مئوية 0-100
    score: float = Field(...)

    skills_match: List[str] = Field(default_factory=list)

    reasoning: str

    faithfulness_score: float = Field(...)


    @validator('score')
    def score_within_percentage_range(cls, v):
        if not (0 <= v <= 100):
            raise ValueError("Score must be between 0 and 100")
        return v


# ---------- API Responses ----------

class RankingResponse(BaseModel):

    job_description: str
    total_candidates_scanned: int
    top_candidates: List[CandidateCard]


class CandidateAnalysisResponse(BaseModel):

    candidate: CandidateCard
    deep_dive: CandidateDeepDive


class CandidateMetadata(BaseModel):

    name: str
    summary: str

    top_skills: List[str] = Field(default_factory=list)

    years_of_experience: Optional[int] = None
    job_title: Optional[str] = None


class MatchResponse(BaseModel):

    total_candidates: int
    top_matches: List[MatchResult]


class CandidateEvaluation(BaseModel):

    candidate_id: str
    summary: str

    faithfulness_score: float = Field(ge=0, le=1)
    relevancy_score: float = Field(ge=0, le=1)


class BatchEvaluationResponse(BaseModel):

    evaluations: List[CandidateEvaluation]
