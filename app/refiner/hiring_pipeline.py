import os
import logging
from typing import Dict, Any, List
from operator import itemgetter
from dotenv import load_dotenv

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

# Local imports from your project
from app.refiner.reranker import rerank_candidates
from app.refiner.scorer import calculate_match_scores
from app.refiner.evaluator import generate_and_evaluate_batch
from app.search_adapter import search_pipeline_to_candidates

# Setup logging for better debugging in production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HiringPipeline")

load_dotenv()

# -----------------------------------------------------
# LLM Configuration
# -----------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.2
)

# -----------------------------------------------------
# Helper Functions for Pipeline Clarity
# -----------------------------------------------------
def format_description(input_data: Dict[str, Any]) -> str:
    """Extracts raw text from JobDescription object or string."""
    desc = input_data.get("description")
    if hasattr(desc, 'description'):
        return desc.description
    return str(desc)

def search_step(input_data: Dict[str, Any]) -> List:
    """Triggers the search phase to fetch initial candidates."""
    description = format_description(input_data)
    logger.info(f"Starting search for JD: {description[:50]}...")
    return search_pipeline_to_candidates(description)

def evaluation_step(input_data: Dict[str, Any]) -> List:
    """Runs the combined explanation and evaluation logic."""
    return generate_and_evaluate_batch(
        description=format_description(input_data),
        job_requirements=input_data.get("job_requirements", []),
        candidates=input_data.get("candidates", [])
    )

# -----------------------------------------------------
# Optimized LCEL Pipeline
# -----------------------------------------------------
# This structure ensures every block has access to original inputs
hiring_pipeline = (
    RunnablePassthrough.assign(
        candidates=RunnableLambda(search_step)
    )
    | RunnablePassthrough.assign(
        # Block 1: Reranking using Local CrossEncoder
        candidates=lambda x: rerank_candidates(format_description(x), x["candidates"])
    )
    | RunnablePassthrough.assign(
        # Block 2: Weighted Scoring Logic
        candidates=lambda x: calculate_match_scores(x["candidates"])
    )
    | RunnablePassthrough.assign(
        # Block 3: AI Deep Analysis (Structured Explanation + Evaluation)
        deep_dives=RunnableLambda(evaluation_step)
    )
)

# -----------------------------------------------------
# Execution Interface
# -----------------------------------------------------
if __name__ == "__main__":
    # Sample Test Case
    test_input = {
        "description": "Senior Frontend Developer with expertise in React, TypeScript, and 5+ years of experience.",
        "job_requirements": ["React", "TypeScript", "Tailwind CSS"]
    }

    print("\n" + "="*50)
    print("🚀 STARTING PROFESSIONAL HIRING PIPELINE")
    print("="*50)

    try:
        final_result = hiring_pipeline.invoke(test_input)

        # Output Results Summary
        print(f"\n✅ Successfully processed {len(final_result['candidates'])} candidates.")
        
        print("\n--- TOP RANKED CANDIDATES ---")
        for i, candidate in enumerate(final_result["candidates"][:3], 1):
            print(f"{i}. {candidate.name} | Score: {candidate.score:.2f} | {candidate.current_title}")
        
        print("\n--- AI EVALUATION (DEEP DIVE) ---")
        for dive in final_result["deep_dives"][:2]:
            status = "TRUSTED" if dive.is_trustworthy else "UNVERIFIED"
            print(f"Candidate ID: {dive.candidate_id} | Status: [{status}]")
            print(f"Reasoning: {dive.explainability.why_match_summary}\n")

    except Exception as e:
        logger.error(f"Pipeline crashed: {str(e)}")
        print(f"\n❌ ERROR: {e}")