"""
Enhanced Search Implementation for Job Description Processing

This module contains the search functionality designed specifically
for job description to CV matching with support for multi-query,
hybrid search, and reranking.
"""
from app.models import JobDescription
from app.vector_store import get_vectorstore
from app.query_expansion import generate_multi_queries
from app.reranker import rerank_candidates
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Import for LangSmith tracing
try:
    from langsmith import traceable
except ImportError:
    # If langsmith is not installed, create a mock decorator
    def traceable(func):
        return func

load_dotenv()

@traceable(name="basic_vector_search", run_type="retriever")
def basic_vector_search(job: JobDescription, k: int = 10):
    """
    Basic vector search using the vector store.
    
    Args:
        job: JobDescription object containing job details (Pydantic model)
        k: Number of results to return
        
    Returns:
        List of matching CV documents with metadata
    """
    # Get the vector store
    vectorstore = get_vectorstore()
    
    # Combine all job details into comprehensive search context
    jd_text = f"""Job Title: {job.title}
Description: {job.description}
Required Skills: {', '.join(job.required_skills) if job.required_skills else 'Not specified'}
Seniority Level: {job.seniority_level or 'Not specified'}
Department: {job.department or 'Not specified'}"""
    
    # Perform basic similarity search
    results = vectorstore.similarity_search(jd_text, k=k)
    
    # Convert to the format expected by the reranking function
    candidates = []
    for doc in results:
        candidates.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "initial_score": getattr(doc, 'score', 0.0)  # Default score if available
        })
    
    return candidates


@traceable(name="get_multi_query_variants", run_type="tool")
def get_multi_query_variants(job: JobDescription, num_queries: int = 3):
    """
    Generates multiple query variations from a JobDescription.
    
    Args:
        job: JobDescription object containing title, description, skills, and seniority level
        num_queries: Number of query variants to generate
         
    Returns:
        List of query strings
    """
    # Combine all job details into comprehensive search context
    jd_text = f"""Job Title: {job.title}
Description: {job.description}
Required Skills: {', '.join(job.required_skills) if job.required_skills else 'Not specified'}
Seniority Level: {job.seniority_level or 'Not specified'}
Department: {job.department or 'Not specified'}"""
    
    # Generate multiple query variations using the existing query expansion module
    queries = generate_multi_queries(jd_text)
    
    # Ensure we have enough queries by adding variations if needed
    while len(queries) < num_queries:
        # Add the original job description as fallback
        queries.append(jd_text)
    
    return queries[:num_queries]


@traceable(name="multi_query_search", run_type="retriever")
def multi_query_search(job: JobDescription, k_per_query: int = 5, total_k: int = 10):
    """
    Performs search using multiple query variants and combines results.
    
    Args:
        job: JobDescription object containing job details (Pydantic model)
        k_per_query: Number of results to fetch per query variant
        total_k: Total number of results to return after deduplication
        
    Returns:
        List of unique matching CV documents with metadata
    """
    # Get multiple query variants
    queries = get_multi_query_variants(job)
    
    # Get vector store
    vectorstore = get_vectorstore()
    
    # Collect results from all queries
    all_results = []
    seen_contents = set()
    
    for query in queries:
        results = vectorstore.similarity_search(query, k=k_per_query)
        for result in results:
            # Deduplicate based on content
            content = result.page_content
            if content not in seen_contents:
                seen_contents.add(content)
                all_results.append(result)
    
    # Limit to total_k results
    results = all_results[:total_k]
    
    # Convert to the format expected by the reranking function
    candidates = []
    for doc in results:
        candidates.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "initial_score": getattr(doc, 'score', 0.0)  # Default score if available
        })
    
    return candidates


@traceable(name="hybrid_search", run_type="retriever")
def hybrid_search(job: JobDescription, k: int = 10):
    """
    Performs hybrid search combining vector and keyword-based approaches.
    
    Args:
        job: JobDescription object containing job details (Pydantic model)
        k: Number of results to return
        
    Returns:
        List of matching CV documents with metadata
    """
    # For now, we'll implement a basic hybrid approach by combining vector search
    # with results from expanded queries
    vector_results = basic_vector_search(job, k=k)
    
    # Generate expanded queries and search with them
    queries = get_multi_query_variants(job)
    
    # Get vector store
    vectorstore = get_vectorstore()
    
    # Collect additional results from expanded queries
    additional_results = []
    seen_contents = set([r["content"] for r in vector_results])
    
    for query in queries:
        results = vectorstore.similarity_search(query, k=k//2)  # Half as many per query
        for result in results:
            content = result.page_content
            if content not in seen_contents:
                seen_contents.add(content)
                additional_results.append({
                    "content": result.page_content,
                    "metadata": result.metadata,
                    "initial_score": getattr(result, 'score', 0.0)
                })
    
    # Combine and limit results
    all_results = vector_results + additional_results
    return all_results[:k]


@traceable(name="search_with_reranking", run_type="chain")
def search_with_reranking(job: JobDescription, top_k: int = 5):
    """
    Complete search pipeline with reranking:
    1. Get candidates from vector store via hybrid search
    2. Get job description processed via Pydantic model
    3. Multi-query generation
    4. Hybrid search (vector + keyword)
    5. Reranking of results
    
    Args:
        job: JobDescription object containing job details (Pydantic model)
        top_k: Number of top results to return after reranking
        
    Returns:
        List of reranked candidate documents with updated scores
    """
    # Get initial candidates using the hybrid search approach
    candidates = hybrid_search(job, k=top_k*2)  # Get more candidates for better reranking
    
    # Perform reranking based on job description
    job_description_text = f"""Title: {job.title}
Description: {job.description}
Skills: {', '.join(job.required_skills)}
Seniority: {job.seniority_level}
Department: {job.department}"""
    
    reranked_candidates = rerank_candidates(
        query=job_description_text,
        candidates=candidates,
        top_k=top_k
    )
    
    return reranked_candidates


@traceable(name="combined_search_pipeline", run_type="chain")
def combined_search_pipeline(job: JobDescription, k: int = 10):
    """
    Combined search pipeline that implements the full flow:
    1. Get candidates from vector store
    2. Process job description with Pydantic model
    3. Multi-query generation from job description
    4. Hybrid search using both vector and keyword approaches
    5. Results returned ready for reranking
    
    Args:
        job: JobDescription object containing job details (using Pydantic model)
        k: Number of results to return
        
    Returns:
        List of matching CV documents with metadata, ready for reranking
    """
    # Use the hybrid search approach to get diversified results
    return hybrid_search(job, k)


# Example usage function
def main():
    # Example JobDescription (for testing purposes)
    job_example = JobDescription(
        title="Software Engineer",
        description="Develop and maintain web applications using modern technologies.",
        required_skills=["Python", "JavaScript", "React"],
        seniority_level="mid",
        department="Engineering"
    )
    
    # Test the basic functionality
    print("Testing search pipeline...")
    results = search_with_reranking(job_example, top_k=3)
    
    print(f"Found {len(results)} candidates after reranking:")
    for i, candidate in enumerate(results, 1):
        print(f"{i}. Score: {candidate['rerank_score']:.3f}")
        print(f"   Content preview: {candidate['content'][:100]}...")
        print()


if __name__ == "__main__":
    main()