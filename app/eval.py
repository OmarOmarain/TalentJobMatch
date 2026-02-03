from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

import os

# Initialize LLM
api_key = os.getenv("GOOGLE_API_KEY")
llm = None
structured_llm = None
eval_prompt = None

class FaithfulnessScore(BaseModel):
    score: float = Field(..., description="Faithfulness score between 0.0 and 1.0")
    reasoning: str = Field(..., description="Explanation of why the candidate matches or does not match")
    skills_found: list[str] = Field(default_factory=list, description="List of skills from JD found in candidate profile")

if api_key:
    try:
        llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-flash")
        structured_llm = llm.with_structured_output(FaithfulnessScore)

        # Prompt for evaluation
        eval_prompt = ChatPromptTemplate.from_template(
            """You are an expert HR evaluator.
            Assess the faithfulness of the match between the Job Description and the Candidate Profile.
            
            Job Description:
            {job_description}
            
            Candidate Profile:
            {candidate_content}
            
            Provide a score (0.0 to 1.0) indicating how well the candidate fits the requirements.
            Provide a concise reasoning.
            List matching skills found in the profile.
            """
        )
    except Exception as e:
        print(f"Failed to initialize Gemini LLM: {e}")

def evaluate_candidate(job_description: str, candidate_content: str) -> dict:
    """
    Evaluates a single candidate against the JD.
    Returns dict with keys: score, reasoning, skills_found
    """
    if not structured_llm or not eval_prompt:
         return {
            "score": 0.5, 
            "reasoning": "Evaluation skipped (GOOGLE_API_KEY missing). returning default score.", 
            "skills_found": []
        }

    try:
        current_chain = eval_prompt | structured_llm
        result = current_chain.invoke({
            "job_description": job_description,
            "candidate_content": candidate_content
        })
        return result.model_dump()
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return {"score": 0.0, "reasoning": "Evaluation failed", "skills_found": []}
