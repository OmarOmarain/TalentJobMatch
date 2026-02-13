from app.models import JobDescription, JobDescriptionRequest
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
from app.core import get_llm

import os

@traceable(name="Parse_JD_Task", run_type="parser")
def parse_job_description_request(request: JobDescriptionRequest) -> JobDescription:
    """
    Enhanced AI Parser: Transforms raw job description text into a structured 
    JobDescription object using Gemini's structured output capabilities.
    """
    
    # 1. Initialize LLM (Using 1.5-flash for speed and reliable extraction)
    # Note: 'gemini-2.0-flash' can also be used if available in your region.
    llm = get_llm(temperature=0.0)
    
    # 2. Bind the LLM to your Pydantic model
    # This forces the AI to return data in the exact format of your JobDescription class
    structured_llm = llm.with_structured_output(JobDescription)
    
    # 3. Enhanced Instructions
    instruction = (
        "You are a professional HR data extractor. "
        "From the provided job description text, extract: "
        "1. A concise professional Job Title. "
        "2. A list of specific technical and soft required_skills. "
        "3. The seniority_level (must be one of: junior, mid, senior, lead). "
        "If the seniority isn't explicit, infer it from years of experience or responsibilities."
    )
    
    try:
        # Construct the final prompt
        full_prompt = f"{instruction}\n\nJob Description Text:\n{request.description}"
        
        # Invoke the structured LLM
        structured_jd = structured_llm.invoke(full_prompt)
        
        # Ensure the description field is populated with the original text if LLM misses it
        if not structured_jd.description:
            structured_jd.description = request.description
            
        return structured_jd
        
    except Exception as e:
        print(f"Extraction failed critical error: {e}")
        # Robust Fallback to keep the system running
        return JobDescription(
            title="Undefined", # Generic title
            description=request.description,
            required_skills=[],
            seniority_level="mid" # Default to mid-level
        )