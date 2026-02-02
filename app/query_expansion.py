from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
api_key = os.getenv("OPENAI_API_KEY")
llm = None
chain = None

if api_key:
    try:
        llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        expansion_prompt = ChatPromptTemplate.from_template(
            """You are an AI job placement assistant. 
            Generate 3 distinct search queries based on the following Job Description (JD) to improve retrieval of relevant candidates.
            Include variations for skills, synonyms, and related job titles.
            
            Job Description:
            {job_description}
            
            Output each query on a new line. Do not number them.
            Do not add any introductory or concluding text, just the queries.
            """
        )
        chain = expansion_prompt | llm | StrOutputParser()
    except Exception as e:
        print(f"Failed to initialize OpenAI LLM: {e}")

def generate_multi_queries(job_description: str) -> list[str]:
    """Generates alternative search queries for a given job description."""
    if not chain:
        print("Warning: OPENAI_API_KEY not found. Skipping query expansion.")
        return [job_description]

    try:
        result = chain.invoke({"job_description": job_description})
        queries = result.strip().split("\n")
        # Clean up empty lines or whitespace and return unique valid queries
        return list(set([q.strip() for q in queries if q.strip()]))
    except Exception as e:
        print(f"Error generating queries: {e}")
        return [job_description]
