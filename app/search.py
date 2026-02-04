"""
Search Implementation for Job Description Processing
"""
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
from app.models import JobDescriptionRequest, JobDescription

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
@traceable(name="get_multi_query_variants", run_type="llm")
@traceable(name="get_multi_query_variants", run_type="llm")
def get_multi_query_variants(job, num_queries: int = 3):

    # ---------- Normalize input ----------
    if isinstance(job, str):

        parsed_job = JobDescription(
            title="Unknown",
            description=job,
            required_skills=[],
            seniority_level=None,
            department=None
        )

    elif isinstance(job, JobDescriptionRequest):

        parsed_job = parse_job_description_request(job)

    elif isinstance(job, JobDescription):

        parsed_job = job

    else:
        raise ValueError(f"Unsupported job type: {type(job)}")

    # ---------- Build JD text ----------
    jd_text = f"""Job Title: {parsed_job.title}
Description: {parsed_job.description}
Required Skills: {', '.join(parsed_job.required_skills) if parsed_job.required_skills else 'Not specified'}
Seniority Level: {parsed_job.seniority_level or 'Not specified'}
Department: {parsed_job.department or 'Not specified'}
"""

    # ---------- Prompt ----------
    multi_query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are a professional technical recruiter.

Generate 3 alternative search queries for CV retrieval.

Job Details: {question}

Output:
VERSION 1: ...
VERSION 2: ...
VERSION 3: ...
"""
    )

    chain = multi_query_prompt | llm
    response = chain.invoke({"question": jd_text})

    generated_queries = response.content.split("\n")

    final_queries = [
        q.split(":")[-1].strip()
        for q in generated_queries
        if ":" in q
    ]

    if not final_queries:
        final_queries = [parsed_job.description]

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
    
    all_docs = vectorstore.get()  # جلب كل الوثائق (أو عينة منها)
    documents = [
        Document(page_content=text, metadata=meta) 
        for text, meta in zip(all_docs['documents'], all_docs['metadatas'])
    ]
    
    keyword_retriever = BM25Retriever.from_documents(documents)
    keyword_retriever.k = k_fetch

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k_fetch})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[0.5, 0.5]
    )

    combined_query = f"{parsed_job.title} " + " ".join(queries)
    hybrid_docs = ensemble_retriever.invoke(combined_query)

    return [{
        "content": doc.page_content,
        "metadata": doc.metadata,
        "initial_score": getattr(doc, 'metadata', {}).get('relevance_score', 0.0)
    } for doc in hybrid_docs[:k_fetch]]

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