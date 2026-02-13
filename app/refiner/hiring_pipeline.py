import os
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

from app.refiner.scorer import process_and_score_candidates
from app.refiner.evaluator import generate_and_evaluate_batch
from app.search_adapter import search_pipeline_to_candidates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HiringPipeline")

load_dotenv()

def format_description(input_data: Dict[str, Any]) -> str:
    desc = input_data.get("description", "")
    if hasattr(desc, 'description'):
        return desc.description
    return str(desc)

def run_hiring_pipeline(input_data: Dict[str, Any]) -> Dict[str, Any]:

    try:
        jd_text = format_description(input_data)
        requirements = input_data.get("job_requirements", [])

        logger.info(f"Step 1: Retrieving candidates...")
        candidates = search_pipeline_to_candidates(jd_text)
        
        if not candidates:
            logger.warning("No candidates found in Vector Store.")
            return {"total_candidates": 0, "top_matches": []}

        logger.info(f"Step 2: Auditing {len(candidates[:2])} candidates via Gemini...")
        deep_dives = generate_and_evaluate_batch(
            description=jd_text,
            job_requirements=requirements,
            candidates=candidates[:6]
        )

        logger.info(f"Step 3: Calculating final scores...")
        final_output = process_and_score_candidates(
            jd_text=jd_text,
            candidates=candidates[:6],
            evaluations=deep_dives
        )

        return {
            "total_candidates": len(candidates),
            "top_matches": final_output
        }

    except Exception as e:
        logger.error(f"Pipeline Critical Failure: {str(e)}")
        raise e

from langchain_core.runnables import RunnableLambda
hiring_pipeline = RunnableLambda(run_hiring_pipeline)

if __name__ == "__main__":
    test_input = {
        "description": "Senior Scrum Master with Agile and Jira expertise.",
        "job_requirements": ["Scrum", "Agile", "Jira"]
    }
    print(run_hiring_pipeline(test_input))