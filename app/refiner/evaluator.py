import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

from app.models import CandidateDeepDive

# --- Gemini Judge ---
judge_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.0  # MUST be deterministic
)


def evaluate_faithfulness(
    explanation: str,
    cv_evidence: str
) -> float:
    """
    Judge whether explanation is strictly supported by CV evidence.
    Returns score between 0.0 and 1.0
    """

    prompt = f"""
You are an impartial evaluator.

TASK:
Evaluate FAITHFULNESS of the explanation.

DEFINITION:
Faithful = every claim is explicitly supported by the evidence.
Unfaithful = hallucination, assumptions, or unsupported claims.

CV EVIDENCE:
{cv_evidence}

EXPLANATION:
{explanation}

Return ONLY a number between 0.0 and 1.0.
"""

    response = judge_llm.invoke([HumanMessage(content=prompt)])
    return float(response.content.strip())


def evaluate_relevancy(
    explanation: str,
    job_description: str
) -> float:
    """
    Judge whether explanation is relevant to the JD.
    """

    prompt = f"""
You are an impartial evaluator.

TASK:
Evaluate RELEVANCY of the explanation to the job description.

JOB DESCRIPTION:
{job_description}

EXPLANATION:
{explanation}

Return ONLY a number between 0.0 and 1.0.
"""

    response = judge_llm.invoke([HumanMessage(content=prompt)])
    return float(response.content.strip())


def evaluate_candidate(
    deep_dive: CandidateDeepDive,
    job_description: str,
    cv_evidence: str,
    faithfulness_threshold: float = 0.75,
    relevancy_threshold: float = 0.70
) -> CandidateDeepDive:
    """
    Full evaluation gate.
    """

    explanation_text = deep_dive.explainability.why_match_summary

    faithfulness = evaluate_faithfulness(explanation_text, cv_evidence)
    relevancy = evaluate_relevancy(explanation_text, job_description)
    deep_dive.relevancy_score = relevancy


    deep_dive.faithfulness_score = faithfulness
    deep_dive.is_trustworthy = (
        faithfulness >= faithfulness_threshold and
        relevancy >= relevancy_threshold
    )

    return deep_dive
