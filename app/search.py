"""
search.py

Enhanced Candidate Retrieval Pipeline with Hybrid Search, Multi-Query Expansion,
and Direct LLM Integration (Google Gemini) for Job Description Processing.
"""

import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from app.models import JobDescription, CandidateCard
from app.vector_store import get_vectorstore

# ------------------------------
# Direct LLM connection (like evaluator)
# ------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.2
)

# ------------------------------
# Multi-Query Generation
# ------------------------------
def generate_multi_queries(jd_text: str, num_queries: int = 3) -> List[str]:
    """
    Generate multiple query variants from a Job Description using LLM.
    """
    prompt = f"""
You are an AI assistant.

TASK:
Generate {num_queries} diverse search query variations from the following job description for candidate retrieval.
Use ONLY the provided information. Do NOT make assumptions.

JOB DESCRIPTION:
{jd_text}

Return the queries as a JSON list of strings.
"""
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        import json
        queries = json.loads(response.content)
    except Exception:
        # fallback if parsing fails
        queries = [jd_text] * num_queries

    while len(queries) < num_queries:
        queries.append(jd_text)

    return queries[:num_queries]

# ------------------------------
# Basic Vector Search
# ------------------------------
def basic_vector_search(job: JobDescription, k: int = 10) -> List[CandidateCard]:
    """
    Perform basic similarity search in vector store.
    """
    vectorstore = get_vectorstore()
    jd_text = f"""Job Title: {job.title}
Description: {job.description}
Required Skills: {', '.join(job.required_skills) if job.required_skills else 'Not specified'}
Seniority Level: {job.seniority_level or 'Not specified'}
Department: {job.department or 'Not specified'}"""

    results = vectorstore.similarity_search(jd_text, k=k)
    candidates = []
    for doc in results:
        candidates.append(CandidateCard(
            candidate_id=doc.metadata.get("candidate_id", "unknown"),
            name=doc.metadata.get("name", "Unknown Candidate"),
            content=doc.page_content,
            matching_skills=[],  # fill later
            years_experience=doc.metadata.get("years_experience", 0),
            current_title=doc.metadata.get("current_title", ""),
            match_score=0.0,
            ai_reasoning_short=""
        ))
    return candidates

# ------------------------------
# Hybrid Search with Multi-Query
# ------------------------------
def hybrid_search(job: JobDescription, k: int = 10) -> List[CandidateCard]:
    """
    Combines vector search with multi-query expansion for diverse results.
    """
    # 1. Basic vector search
    base_results = basic_vector_search(job, k=k)

    # 2. Generate multi-query variants
    jd_text = f"{job.title} {job.description}"
    queries = generate_multi_queries(jd_text, num_queries=3)

    # 3. Collect additional results
    vectorstore = get_vectorstore()
    additional_results = []
    seen_contents = set([c.content for c in base_results])

    for query in queries:
        results = vectorstore.similarity_search(query, k=k//2)
        for doc in results:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                additional_results.append(CandidateCard(
                    candidate_id=doc.metadata.get("candidate_id", "unknown"),
                    name=doc.metadata.get("name", "Unknown Candidate"),
                    content=doc.page_content,
                    matching_skills=[],
                    years_experience=doc.metadata.get("years_experience", 0),
                    current_title=doc.metadata.get("current_title", ""),
                    match_score=0.0,
                    ai_reasoning_short=""
                ))

    all_candidates = base_results + additional_results
    return all_candidates[:k]

# ------------------------------
# Example usage (local testing)
# ------------------------------
if __name__ == "__main__":
    from app.models import JobDescription

    job_example = JobDescription(
        title="UI/UX Designer",
        description="Looking for a skilled UI/UX designer with Adobe XD and Photoshop experience.",
        required_skills=["UI/UX design", "Adobe XD", "Photoshop"],
        seniority_level="mid",
        department="Design"
    )

    results = hybrid_search(job_example, k=5)
    print(f"Found {len(results)} candidates:")
    for i, c in enumerate(results, 1):
        print(f"{i}. {c.name} - {c.content[:100]}...")
