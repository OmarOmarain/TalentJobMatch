import logging
import time
from typing import Dict, Any, List

from app.refiner.scorer import process_and_score_candidates
from app.refiner.evaluator import generate_and_evaluate_batch
from app.search_adapter import search_pipeline_to_candidates


logger = logging.getLogger("HiringPipeline")

MAX_AUDIT_CANDIDATES = 6


def format_description(input_data: Dict[str, Any]) -> str:
    desc = input_data.get("description", "")
    return str(desc).strip()


async def run_hiring_pipeline(input_data: Dict[str, Any]) -> Dict[str, Any]:

    start_time = time.time()

    jd_text = format_description(input_data)
    requirements = input_data.get("job_requirements", [])

    logger.info("Retrieving candidates...")
    candidates = search_pipeline_to_candidates(jd_text)

    if not candidates:
        return {"total_candidates": 0, "top_matches": []}

    audit_candidates = candidates[:MAX_AUDIT_CANDIDATES]

    logger.info("Running Async Gemini Audit...")

    deep_dives = await generate_and_evaluate_batch(
        description=jd_text,
        job_requirements=requirements,
        candidates=audit_candidates
    )

    logger.info("Scoring...")

    final_output = process_and_score_candidates(
        jd_text=jd_text,
        candidates=audit_candidates,
        evaluations=deep_dives
    )

    logger.info(f"Pipeline finished in {round(time.time()-start_time,2)}s")

    return {
        "total_candidates": len(candidates),
        "top_matches": final_output
    }
