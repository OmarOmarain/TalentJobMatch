from dotenv import load_dotenv
import os

load_dotenv()  # يقرأ .env

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.refiner.reranker import rerank_candidates # Reranking
from app.refiner.scorer import calculate_match_scores # Final scoring
from app.refiner.explainer import generate_explanations # Explainability
from app.refiner.evaluator import evaluate_candidate  # Faithfulness & relevancy


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.2
)



# =====================================================
# BLOCK 1 — RERANK
# =====================================================

rerank_block = RunnableLambda(
    lambda x: {
        **x,
        "candidates": rerank_candidates(
            x["job_description"],
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
        "candidates": calculate_match_scores(
            x["candidates"]
        )
    }
)


# =====================================================
# BLOCK 3 — EXPLAIN
# =====================================================

explain_block = RunnableLambda(
    lambda x: {
        **x,
        "deep_dives": generate_explanations(
            x["job_description"],
            x["job_requirements"],
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

        evaluated.append(
            evaluate_candidate(
                deep_dive=deep_dive,
                job_description=x["job_description"],
                cv_evidence=str(x["candidates"])
            )
        )

    return {
        **x,
        "deep_dives": evaluated
    }


evaluate_block = RunnableLambda(evaluate_all)


# =====================================================
# FULL PIPELINE
# =====================================================

hiring_pipeline = (
    RunnablePassthrough()
    | rerank_block
    | score_block
    | explain_block
    | evaluate_block
)


if __name__ == "__main__":
    from app.models import CandidateCard, IdentifiedSkill  # أو SkillChip حسب مشروعك

    # ----- Mock JD + Requirements -----
    jd = "Looking for a Frontend Engineer skilled in Vue with 5 years experience."
    requirements = ["Vue"]


    # ----- Mock candidates -----
    candidates = [
        CandidateCard(
            candidate_id="1",
            name="Alice",
            avatar_url=None,
            current_title="Frontend Developer",
            company="TechCorp",
            years_experience=3,
            seniority_level="Mid",
            location="Cairo",
            score=0.0,
            skills_match=["React", "Vue"],
            ai_reasoning_short="",
        ),
        CandidateCard(
            candidate_id="2",
            name="Bob",
            avatar_url=None,
            current_title="Frontend Engineer",
            company="Innovate",
            years_experience=5,
            seniority_level="Senior",
            location="Cairo",
            score=0.0,
            skills_match=["Vue"],
            ai_reasoning_short="",
        ),
    ]

    # ----- Run pipeline -----
    result = hiring_pipeline.invoke({
        "job_description": jd,
        "job_requirements": requirements,
        "candidates": candidates
    })

    # ----- Print results -----
    print("\n=== Candidates After Pipeline ===\n")
    for c in result["candidates"]:
        print(f"{c.name}: score={c.score}, skills={c.skills_match}, reasoning={c.ai_reasoning_short}")

    print("\n=== Deep Dives / Explanations ===\n")
    for d in result["deep_dives"]:
        print(f"{d.candidate_id}: faithfulness={d.faithfulness_score}, relevancy={d.relevancy_score}, trustworthy={d.is_trustworthy}")
        print(f"Summary: {d.explainability.why_match_summary}\n")

