import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from app.models import CandidateDeepDive


# --- Gemini Judge ---
judge_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.0
)


# -------------------------
# Helper: Extract float safely
# -------------------------
def extract_score(text: str) -> float:
    """
    Extract first float between 0 and 1 from LLM response.
    Prevents crashes if model returns extra text.
    """

    matches = re.findall(r"0\.\d+|1\.0+|1|0", text)

    if matches:
        return float(matches[-1])  # آخر رقم غالباً هو النتيجة

    return 0.0


# -------------------------
# Faithfulness
# -------------------------
def evaluate_faithfulness(
    explanation: str,
    cv_evidence: str
) -> float:

    prompt = f"""
You are an impartial evaluator.

TASK:
Evaluate FAITHFULNESS of the explanation.

Faithful means:
- Every claim is supported by evidence.

Return ONLY ONE NUMBER between 0.0 and 1.0.
Do not explain.
Do not write anything else.

CV EVIDENCE:
{cv_evidence}

EXPLANATION:
{explanation}
"""

    response = judge_llm.invoke([HumanMessage(content=prompt)])
    return extract_score(response.content)


# -------------------------
# Relevancy
# -------------------------
def evaluate_relevancy(
    explanation: str,
    job_description: str
) -> float:

    prompt = f"""
You are an impartial evaluator.

TASK:
Evaluate RELEVANCY of the explanation to the job description.

Return ONLY ONE NUMBER between 0.0 and 1.0.
Do not explain.

JOB DESCRIPTION:
{job_description}

EXPLANATION:
{explanation}
"""

    response = judge_llm.invoke([HumanMessage(content=prompt)])
    return extract_score(response.content)


# -------------------------
# Full Candidate Evaluation
# -------------------------
def evaluate_candidate(
    deep_dive: CandidateDeepDive,
    job_description: str,
    cv_evidence: str,
    faithfulness_threshold: float = 0.75,
    relevancy_threshold: float = 0.70
) -> CandidateDeepDive:

    explanation_text = deep_dive.explainability.why_match_summary

    faithfulness = evaluate_faithfulness(
        explanation_text,
        cv_evidence
    )

    relevancy = evaluate_relevancy(
        explanation_text,
        job_description
    )

    deep_dive.faithfulness_score = faithfulness
    deep_dive.relevancy_score = relevancy

    deep_dive.is_trustworthy = (
        faithfulness >= faithfulness_threshold
        and relevancy >= relevancy_threshold
    )

    return deep_dive
