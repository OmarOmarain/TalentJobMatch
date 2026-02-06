from dotenv import load_dotenv
import os

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from app.refiner.reranker import rerank_candidates
from app.refiner.scorer import calculate_match_scores
from app.refiner.explainer import generate_explanations
from app.refiner.evaluator import evaluate_candidate

from app.search import combined_search_pipeline
from app.search_adapter import search_pipeline_to_candidates

load_dotenv()

# =====================================================
# LLM
# =====================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.2
)

# =====================================================
# BLOCK 0 — SEARCH
# =====================================================

search_block = RunnableLambda(
    lambda x: {
        **x,
        "candidates": search_pipeline_to_candidates(x["description"])
    }
)

# =====================================================
# BLOCK 1 — RERANK
# =====================================================

rerank_block = RunnableLambda(
    lambda x: {
        **x,
        "candidates": rerank_candidates(
            x["description"].description if hasattr(x["description"], 'description') else x["description"],
            x["candidates"]
        )
    }
)

# =====================================================
# BLOCK 2 — SCORE
# =====================================================

score_block = RunnableLambda(
    lambda x: {
        **x,
        "candidates": calculate_match_scores(x["candidates"])
    }
)

# =====================================================
# BLOCK 3 — EXPLAIN
# =====================================================

explain_block = RunnableLambda(
    lambda x: {
        **x,
        "deep_dives": generate_explanations(
            x["description"].description if hasattr(x["description"], 'description') else x["description"],
            x.get("job_requirements", []),
            x["candidates"]
        )
    }
)

# =====================================================
# BLOCK 4 — EVALUATE
# =====================================================

def evaluate_all(x):
    evaluated = []
    
    for deep_dive in x["deep_dives"]:
        description_text = (
            x["description"].description 
            if hasattr(x["description"], 'description') 
            else str(x["description"])
        )
        
        evaluated.append(
            evaluate_candidate(
                deep_dive=deep_dive,
                description=description_text,
                cv_evidence=str(x["candidates"])
            )
        )
    
    return {**x, "deep_dives": evaluated}

evaluate_block = RunnableLambda(evaluate_all)  # ✅ أضف هذا السطر!

# =====================================================
# FULL PIPELINE
# =====================================================

hiring_pipeline = (
    RunnablePassthrough()
    | search_block
    | rerank_block
    | score_block
    | explain_block
    | evaluate_block
)

# =====================================================
# LOCAL TEST
# =====================================================

if __name__ == "__main__":
    jd = "Looking for a Frontend Engineer skilled in Vue with 5 years experience."
    requirements = ["Vue"]
    
    result = hiring_pipeline.invoke({
        "description": jd,
        "job_requirements": requirements
    })
    
    print("\n=== Candidates After Pipeline ===\n")
    for c in result["candidates"]:
        print(f"{c.name}: score={c.score}, skills={c.skills_match}")
    
    print("\n=== Deep Dives / Explanations ===\n")
    for d in result["deep_dives"]:
        print(f"{d.candidate_id}: faithfulness={d.faithfulness_score}")