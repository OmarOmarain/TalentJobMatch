"""
Search Implementation for Job Description Processing
"""
from app.models import JobDescription, JobDescriptionRequest
from app.vector_store import get_vectorstore
from app.parser import parse_job_description_request
import os
from dotenv import load_dotenv
from langsmith import traceable
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing import List
from langchain_classic.retrievers import BM25Retriever,EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
@traceable(name="get_multi_query_variants", run_type="llm")
def get_multi_query_variants(job, num_queries: int = 3):
    """Generates multiple query variations from either JobDescription or JobDescriptionRequest."""
    # Handle both JobDescription and JobDescriptionRequest inputs
    if isinstance(job, JobDescriptionRequest):
        # Parse the request into structured JobDescription
        parsed_job = parse_job_description_request(job)
    elif isinstance(job, JobDescription):
        # Already structured, use as-is
        parsed_job = job
    else:
        raise ValueError(f"Expected JobDescription or JobDescriptionRequest, got {type(job)}")
    
    # Combine all job details into comprehensive search context
    jd_text = f"""Job Title: {parsed_job.title}
Description: {parsed_job.description}
Required Skills: {', '.join(parsed_job.required_skills) if parsed_job.required_skills else 'Not specified'}
Seniority Level: {parsed_job.seniority_level or 'Not specified'}
Department: {parsed_job.department or 'Not specified'}
"""
    
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
  

    chain = multi_query_prompt | llm 

    response = chain.invoke({"question": jd_text})
    
   
    generated_queries = response.content.split("\n")
    
    final_queries = [q.split(":")[-1].strip() for q in generated_queries if ":" in q]

    if not final_queries:
        final_queries = [f"{parsed_job.title} {', '.join(parsed_job.required_skills[:5])}"]

    return final_queries[:num_queries]

@traceable(name="hybrid_search", run_type="retriever")
def hybrid_search(parsed_job: JobDescription, queries: List[str], k_fetch: int = 15):
    """
    Performs TRUE Hybrid Search:
    1. Semantic Search (Vector)
    2. Keyword Search (BM25)
    3. Merges them using Reciprocal Rank Fusion (RRF)
    """
    vectorstore = get_vectorstore()
    
    # Optimize by using the vector store's native retrieval first
    # This avoids loading all documents into memory
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k_fetch})
    
    # Create a combined query from the job title and all query variants
    combined_query = f"{parsed_job.title} " + " ".join(queries)
    
    # Get vector search results first
    vector_results = vector_retriever.invoke(combined_query)
    
    # Use the same set of documents for both retrievers to avoid reloading
    keyword_retriever = BM25Retriever.from_documents(vector_results)
    keyword_retriever.k = k_fetch

    # Use individual retrievers and combine results manually
    vector_results = vector_retriever.invoke(combined_query)
    keyword_results = keyword_retriever.invoke(combined_query)

    # Deduplicate results while preserving order and scores
    seen_content = set()
    final_results = []
    
    # Add vector results first (typically more semantically relevant)
    for doc in vector_results:
        content = doc.page_content
        if content not in seen_content:
            seen_content.add(content)
            final_results.append(doc)
    
    # Add keyword results that aren't already in the list
    for doc in keyword_results:
        content = doc.page_content
        if content not in seen_content:
            seen_content.add(content)
            final_results.append(doc)

    return [{
        "content": doc.page_content,
        "metadata": doc.metadata,
        "initial_score": getattr(doc, 'metadata', {}).get('relevance_score', 0.0)
    } for doc in final_results[:k_fetch]]

@traceable(name="combined_search_pipeline", run_type="chain")
def combined_search_pipeline(job, k: int = 10):
    if isinstance(job, JobDescriptionRequest):
        parsed_job = parse_job_description_request(job)
    else:
        parsed_job = job
    queries = get_multi_query_variants(parsed_job, num_queries=3)
    K_FETCH = 15
    hybrid_results = hybrid_search(parsed_job, queries, k_fetch=K_FETCH)

    all_results = []
    seen_contents = set()

    for res in hybrid_results:
        content = res['content'] 
        if content not in seen_contents:
            seen_contents.add(content)
            all_results.append({
                "content": content,
                "metadata": res['metadata'],
                "score": res.get('initial_score', 0.0)
            })

    return all_results[:k]