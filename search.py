"""
Enhanced Multi-Query Implementation for Job Description Processing

This module contains the enhanced multi-query functionality 
designed specifically for job description to CV matching.
"""
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from app.core import vectorstore, llm

langsmith.init()


def get_multi_query_retriever(job_description: str):
    """
    Generates multiple query variations from a job description and retrieves matching CVs
    
    Args:
        job_description: The job description text to generate queries from
         
    Returns:
        List of matching CV documents ranked by relevance
    """
    # Base retriever with k parameter for consistency
    base_retriever = vectorstore.as_retriever(search_kwargs={"k":30})
    
    multi_query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are a professional technical recruiter with expertise in CV matching.
        Your task is to generate 3 different alternative versions of the following job description 
        to help find the best candidate CVs in a vector database.
        
        Focus on:
        - Technical skills and technologies required
        - Core job responsibilities
        - Experience level and seniority
        
        Original Job Description: {question}
        
        Generate 3 alternative versions. Output format:
        VERSION 1: [first alternative]
        VERSION 2: [second alternative]
        VERSION 3: [third alternative]"""
    )

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        prompt=multi_query_prompt
    )
    
    # Get and return results
    results = multi_query_retriever.invoke({"query": job_description})
    return results

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
    #output structure
    #vectore store type 
    