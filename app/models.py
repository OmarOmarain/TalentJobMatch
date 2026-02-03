from typing import List, Optional
from pydantic import BaseModel, Field

class JobDescription(BaseModel):
    title: str = Field(..., description="Job Title")
    description: str = Field(..., description="Full job description text")
    required_skills: List[str] = Field(default_factory=list, description="List of required skills")

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
