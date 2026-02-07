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
from app.performance_monitor import timing_decorator, perf_monitor

app = FastAPI(title="Talent Job Matching API", version="1.0",debug=True)

@app.get("/")
def read_root():
    return {"message": "Welcome to Talent Job Matching API. Server is running!"}

@app.post("/api/v1/match/candidate", response_model=MatchResponse)
@timing_decorator
async def match_candidates(job: JobDescriptionRequest):
    import time
    start_time = time.time()
    
    job_full = JobDescription(
        title="Unknown",
        description=job.description,
        required_skills=[]
    )
    
    pipeline_start = time.time()
    result = hiring_pipeline.invoke({
        "description": job_full,
        "job_requirements": []
    })
    pipeline_end = time.time()
    perf_monitor.record_metric("hiring_pipeline_execution", pipeline_end - pipeline_start)
    
    candidates = result.get("candidates", [])
    deep_dives = result.get("deep_dives", [])
    
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
    
    end_time = time.time()
    perf_monitor.record_metric("match_candidates_total", end_time - start_time)
    
    # Print performance report after processing
    perf_monitor.print_report()
    
    return MatchResponse(
        total_candidates=len(candidates),
        top_matches=final_matches
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)