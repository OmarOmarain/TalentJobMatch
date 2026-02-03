from typing import List, Optional
from pydantic import BaseModel, Field

class Candidate(BaseModel):
    id: str = Field(..., description="Unique identifier for the candidate")
    name: str = Field(..., description="Full name of the candidate")
    email: Optional[str] = Field(None, description="Email address")
    skills: List[str] = Field(default_factory=list, description="List of technical skills")
    experience_years: int = Field(0, description="Total years of experience")
    summary: str = Field(..., description="Brief professional summary")
    raw_text: str = Field(..., description="Full text from the resume")

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


class JobDescription(BaseModel):
    title: str = Field(..., min_length=3, max_length=200, description="Job Title")
    description: str = Field(..., min_length=50, description="Full job description text")
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

class MatchResult(BaseModel):
    candidate_id: str
    name: str
    score: float = Field(..., description="Overall match score (0-1)")
    skills_match: List[str] = Field(default_factory=list, description="Skills found in both JD and Candidate")
    reasoning: str = Field(..., description="AI explanation for the match")
    faithfulness_score: float = Field(..., description="Faithfulness score of the explanation")

class MatchResponse(BaseModel):
    total_candidates: int
    top_matches: List[MatchResult]
