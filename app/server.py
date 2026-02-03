import os
from dotenv import load_dotenv
import langsmith

# Load environment variables before importing other modules
load_dotenv()

# Initialize LangSmith tracing
langsmith.init()

from fastapi import FastAPI, HTTPException
from app.models import JobDescription, MatchResponse, MatchResult
from app.search_pipeline import search_pipeline
from app.reranker import rerank_candidates
from app.eval import evaluate_candidate
import uvicorn

app = FastAPI(title="Talent Job Matching API", version="1.0")

@app.get("/")
def read_root():
    return {"message": "Welcome to Talent Job Matching API. Server is running!"}

@app.post("/api/v1/match/candidate", response_model=MatchResponse)
async def match_candidates(job: JobDescription):
    """
    Endpoint to match candidates against a Job Description.
    """
    jd_text = f"{job.title}\n{job.description}\nKeywords: {', '.join(job.required_skills)}"
    
    # 1. Hybrid Search (Vector + BM25)
    # Get more candidates initially to rerank later
    print(f"Searching for: {job.title}")
    initial_matches = search_pipeline.hybrid_search(jd_text, k=10)
    
    if not initial_matches:
        return MatchResponse(total_candidates=0, top_matches=[])
        
    print(f"Found {len(initial_matches)} initial matches.")

    # 2. Reranking (Cross-Encoder)
    # Refine to top 5
    reranked = rerank_candidates(jd_text, initial_matches, top_k=5)
    print("Reranking complete.")

    # 3. Faithfulness & Final Formatting
    final_matches = []
    
    for cand in reranked:
        # Extract metadata if available
        meta = cand.get('metadata', {})
        cand_content = cand.get('content', '')
        
        # Evaluate
        eval_result = evaluate_candidate(jd_text, cand_content)
        
        match_result = MatchResult(
            candidate_id=meta.get('doc_id', 'unknown'),
            name=meta.get('source', 'Unknown Candidate'), # Using source as name for now if not structured
            score=cand.get('rerank_score', 0.0),
            skills_match=eval_result['skills_found'],
            reasoning=eval_result['reasoning'],
            faithfulness_score=eval_result['score']
        )
        final_matches.append(match_result)

    return MatchResponse(
        total_candidates=len(initial_matches),
        top_matches=final_matches
    )

if __name__ == "__main__":
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)
