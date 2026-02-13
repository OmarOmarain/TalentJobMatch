import logging
import time
from typing import Dict, Any, List

from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda

from app.refiner.scorer import process_and_score_candidates
from app.refiner.evaluator import generate_and_evaluate_batch
from app.search_adapter import search_pipeline_to_candidates


# ---------- Config ----------
MAX_AUDIT_CANDIDATES = 6

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HiringPipeline")

load_dotenv()


# ---------- Helpers ----------

def format_description(input_data: Dict[str, Any]) -> str:
    desc = input_data.get("description", "")

    if hasattr(desc, "description"):
        desc = desc.description

    return str(desc).strip()


def normalize_requirements(reqs: Any) -> List[str]:

    if not reqs:
        return []

    if isinstance(reqs, str):
        return [reqs]

    if isinstance(reqs, list):
        return [str(r).strip() for r in reqs if r]

    return []


# ---------- Main Pipeline ----------

def run_hiring_pipeline(input_data: Dict[str, Any]) -> Dict[str, Any]:

    start_time = time.time()

    try:
        # ---------- Step 0: Prepare Inputs ----------
        jd_text = format_description(input_data)
        requirements = normalize_requirements(input_data.get("job_requirements"))

        if not jd_text:
            logger.warning("Empty Job Description received.")
            return {"total_candidates": 0, "top_matches": []}

        logger.info("Pipeline started")

        # ---------- Step 1: Candidate Retrieval ----------
        t0 = time.time()

        candidates = search_pipeline_to_candidates(jd_text)

        retrieval_time = round(time.time() - t0, 2)

        if not candidates:
            logger.warning("No candidates retrieved from search pipeline.")
            return {"total_candidates": 0, "top_matches": []}

        logger.info(f"Retrieved {len(candidates)} candidates in {retrieval_time}s")

        # ---------- Limit Candidates ----------
        audit_candidates = candidates[:MAX_AUDIT_CANDIDATES]

        # ---------- Step 2: LLM Audit ----------
        t1 = time.time()

        deep_dives = generate_and_evaluate_batch(
            description=jd_text,
            job_requirements=requirements,
            candidates=audit_candidates
        )

        audit_time = round(time.time() - t1, 2)

        logger.info(
            f"LLM Audit completed for {len(deep_dives)} candidates in {audit_time}s"
        )

        # ---------- Step 3: Scoring ----------
        t2 = time.time()

        final_output = process_and_score_candidates(
            jd_text=jd_text,
            candidates=audit_candidates,
            evaluations=deep_dives
        )

        scoring_time = round(time.time() - t2, 2)

        logger.info(f"Scoring completed in {scoring_time}s")

        total_time = round(time.time() - start_time, 2)

        logger.info(f"Pipeline completed successfully in {total_time}s")

        return {
            "total_candidates": len(candidates),
            "audited_candidates": len(audit_candidates),
            "top_matches": final_output
        }

    except Exception as e:
        logger.exception("Pipeline Critical Failure")
        raise e


# ---------- LangChain Runnable ----------
hiring_pipeline = RunnableLambda(run_hiring_pipeline)


# ---------- Local Test ----------
if __name__ == "__main__":

    test_input = {
        "description": "Senior Scrum Master with Agile and Jira expertise.",
        "job_requirements": ["Scrum", "Agile", "Jira"]
    }

    print(run_hiring_pipeline(test_input))
