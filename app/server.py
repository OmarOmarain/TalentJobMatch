import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI
from typing import List
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()

from app.models import (
    JobDescriptionRequest,
    MatchResponse,
    MatchResult
)

from app.refiner.hiring_pipeline import run_hiring_pipeline
from app.performance_monitor import timing_decorator, perf_monitor

app = FastAPI(title="Talent Job Matching API", version="1.0",debug=True)
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=False,allow_methods=["*"],allow_headers=["*"])

@app.get("/")
def read_root():
    return {"message": "Welcome to Talent Job Matching API. Server is running!"}

@app.post("/api/v1/match/candidate", response_model=MatchResponse)
@timing_decorator
async def match_candidates(job: JobDescriptionRequest):
    start_time = time.time()
    
    pipeline_start = time.time()
    
    result = run_hiring_pipeline({
        "description": job.description,
        "job_requirements": [] 
    })
    
    pipeline_end = time.time()
    perf_monitor.record_metric("hiring_pipeline_execution", pipeline_end - pipeline_start)
    
    total_found = result.get("total_candidates", 0)
    top_matches_data = result.get("top_matches", [])

    final_matches: List[MatchResult] = []
    for item in top_matches_data:
        final_matches.append(
            MatchResult(
                candidate_id=item["candidate_id"],
                name=item["name"],
                score=item["score"],
                skills_match=item["skills_match"],
                reasoning=item["reasoning"],
                faithfulness_score=item["faithfulness_score"]
            )
        )
    
    end_time = time.time()
    perf_monitor.record_metric("match_candidates_total", end_time - start_time)
    
    perf_monitor.print_report()
    
    return MatchResponse(
        total_candidates=total_found,
        top_matches=final_matches
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)
