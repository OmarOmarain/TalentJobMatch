"""
Enhanced Multi-Query Implementation for Job Description Processing

This module contains the enhanced multi-query functionality 
designed specifically for job description to CV matching.
"""
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from app.vector_store import vectorstore
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo") if api_key else None
from app.models import JobDescription


def get_multi_query_retriever(job: JobDescription):
    """
    Generates multiple query variations from a JobDescription and retrieves matching CVs.
    Uses rich context (title, skills, seniority) for better query generation.
    
    Args:
        job: JobDescription object containing title, description, skills, and seniority level
         
    Returns:
        List of matching CV documents ranked by relevance
    """
    # Combine all job details into comprehensive search context
    jd_text = f"""Job Title: {job.title}
Description: {job.description}
Required Skills: {', '.join(job.required_skills) if job.required_skills else 'Not specified'}
Seniority Level: {job.seniority_level or 'Not specified'}
Department: {job.department or 'Not specified'}"""
    
    # Base retriever with k parameter for consistency
    base_retriever = vectorstore.as_retriever(search_kwargs={"k":30})
    
    multi_query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are a professional technical recruiter with expertise in CV matching.
        Your task is to generate 3 different alternative search queries based on the following job details
        to help find the best candidate CVs in a vector database.
        
        Focus on:
        - Technical skills and technologies required
        - Core job responsibilities
        - Experience level and seniority
        
        Job Details: {question}
        
        Generate 3 alternative search queries. Output format:
        VERSION 1: [first query]
        VERSION 2: [second query]
        VERSION 3: [third query]"""
    )

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        prompt=multi_query_prompt
    )
    
    # Get and return results with error handling
    try:
        results = multi_query_retriever.invoke({"question": jd_text})
        return results if results else []
    except Exception as e:
        print(f"Error in multi-query retrieval: {e}")
        return []

def get_ensemble_retriever():
    """
    Creates an ensemble retriever combining BM25 and vector store retrievers
    
    Returns:
        An EnsembleRetriever instance
    """
    bm25_retriever = BM25Retriever.from_documents(vectorstore.get_documents())
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k":15})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]  # Adjust weights as needed
    )
    
    return ensemble_retriever

def combined_search_pipeline(job: JobDescription, k: int = 10):
    """
    Combined search pipeline that implements the full flow:
    1. Multi-query generation from job description
    2. Hybrid search using both vector and keyword (BM25) approaches
    3. Returns results ready for reranking
    
    Args:
        job: JobDescription object containing job details
        k: Number of results to return
        
    Returns:
        List of matching CV documents with metadata, ready for reranking
    """
    # Step 1: Use multi-query approach to generate varied search queries
    multi_query_results = get_multi_query_retriever(job)
    
    # Step 2: Use ensemble/hybrid approach for broader coverage
    ensemble_retriever = get_ensemble_retriever()
    
    # Combine job details for search
    jd_text = f"""Job Title: {job.title}
Description: {job.description}
Required Skills: {', '.join(job.required_skills) if job.required_skills else 'Not specified'}
Seniority Level: {job.seniority_level or 'Not specified'}
Department: {job.department or 'Not specified'}"""
    
    # Get results from ensemble retriever
    ensemble_results = ensemble_retriever.invoke(input=jd_text)
    
    # Combine and deduplicate results from both approaches
    all_results = list(set(multi_query_results + ensemble_results))
    
    # Convert to the format expected by the reranking function
    candidates = []
    for doc in all_results[:k]:
        candidates.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "initial_score": getattr(doc, 'score', 0.0)  # Default score if available
        })
    
    return candidates

#output structure
#vector store type
    