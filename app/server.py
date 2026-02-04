import os
from dotenv import load_dotenv
from fastapi import FastAPI
from typing import List

# Load environment variables
load_dotenv()

# Import models
from app.models import (
    JobDescription,
    MatchResponse,
    MatchResult,
    CandidateCard,
    CandidateDeepDive
)

# Import pipeline modules
from app.search import hybrid_search                # Retrieval
from app.refiner.reranker import rerank_candidates # Reranking
from app.refiner.scorer import calculate_match_scores # Final scoring
from app.refiner.explainer import generate_explanations # Explainability
from app.refiner.evaluator import evaluate_candidate  # Faithfulness & relevancy

app = FastAPI(title="Talent Job Matching API", version="1.0")


@app.get("/")
def read_root():
    return {"message": "Welcome to Talent Job Matching API. Server is running!"}


@app.post("/api/v1/match/candidate", response_model=MatchResponse)
async def match_candidates(job: JobDescription):
    """
    Endpoint to match candidates against a Job Description.
    Full pipeline:
    1. Hybrid Search (vector + multi-query)
    2. Reranking (Cross-Encoder)
    3. Final scoring
    4. Explainability generation
    5. Evaluation (faithfulness + relevancy)
    """

    # -----------------------------
    # 1️⃣ Retrieval
    # -----------------------------
    initial_candidates: List[CandidateCard] = hybrid_search(job, k=10)
    if not initial_candidates:
        return MatchResponse(total_candidates=0, top_matches=[])

    # -----------------------------
    # 2️⃣ Reranking
    # -----------------------------
    jd_text = job.description  # ✅ نص كامل فقط
    reranked_candidates = rerank_candidates(
        job_description=jd_text,
        candidates=initial_candidates
    )

    # -----------------------------
    # 3️⃣ Final Scoring
    # -----------------------------
    scored_candidates = calculate_match_scores(reranked_candidates)

    # -----------------------------
    # 4️⃣ Explainability
    # -----------------------------
    job_requirements: List[str] = []  # لم يعد هناك skills
    explanations: List[CandidateDeepDive] = generate_explanations(
        job_description=jd_text,
        job_requirements=job_requirements,
        candidates=scored_candidates
    )

    # -----------------------------
    # 5️⃣ Evaluation (faithfulness & relevancy)
    # -----------------------------
    final_matches: List[MatchResult] = []
    for cand, deep_dive in zip(scored_candidates, explanations):
        # Use candidate content as CV evidence
        cv_text = getattr(cand, "content", "") or ""
        evaluated = evaluate_candidate(
            deep_dive=deep_dive,
            job_description=jd_text,
            cv_evidence=cv_text
        )

        final_matches.append(MatchResult(
            candidate_id=cand.candidate_id,
            name=cand.name,
            score=cand.match_score,
            skills_match=[s.name for s in cand.matching_skills],
            reasoning=cand.ai_reasoning_short,
            faithfulness_score=evaluated.faithfulness_score
        ))

    return MatchResponse(
        total_candidates=len(initial_candidates),
        top_matches=final_matches
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)
