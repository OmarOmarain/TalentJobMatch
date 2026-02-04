import os
from dotenv import load_dotenv
from fastapi import FastAPI
from typing import List

load_dotenv()

from app.models import (
    JobDescription,
    JobDescriptionRequest,
    MatchResponse,
    MatchResult
)


from app.refiner.hiring_pipeline import hiring_pipeline

app = FastAPI(title="Talent Job Matching API", version="1.0")


@app.get("/")
def read_root():
    return {"message": "Welcome to Talent Job Matching API. Server is running!"}


@app.post("/api/v1/match/candidate", response_model=MatchResponse)
async def match_candidates(job: JobDescriptionRequest):   # ✅ كان ناقص :

    # ✅ تحويل request إلى JobDescription كامل
    job_full = JobDescription(
        title="Unknown",
        description=job.description,
        required_skills=[]
    )

    # ✅ تشغيل الـ pipeline
    result = hiring_pipeline.invoke({
        "job": job_full,
        "job_description": job_full.description,
        "job_requirements": job_full.required_skills
    })

    candidates = result["candidates"]
    deep_dives = result["deep_dives"]

    # ✅ بناء الاستجابة
    final_matches: List[MatchResult] = []

    for cand, deep_dive in zip(candidates, deep_dives):
        final_matches.append(
            MatchResult(
                candidate_id=cand.candidate_id,
                name=cand.name,
                score=cand.score,
                skills_match=cand.skills_match,
                reasoning=cand.ai_reasoning_short,
                faithfulness_score=deep_dive.faithfulness_score
            )
        )

    return MatchResponse(
        total_candidates=len(candidates),
        top_matches=final_matches
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)