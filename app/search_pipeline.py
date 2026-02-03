from typing import List
from app.vector_store import get_vectorstore
from app.bm25_index import BM25Index
from app.query_expansion import generate_multi_queries

class SearchPipeline:
    def __init__(self):
        # In a real app, you might lazily load the BM25 index from all docs in Chroma
        # For this template, we assume it's rebuilt on startup or request (expensive for large data)
        self.bm25_index = None

    def _build_bm25(self):
        # Fetch all documents from Chroma (this is just a placeholder logic)
        # In production, use a more efficient way or a persistent keyword store
        vectorstore = get_vectorstore()
        all_docs = vectorstore.get()
        if all_docs and all_docs['documents']:
            self.bm25_index = BM25Index(all_docs['documents'])

    def hybrid_search(self, job_description: str, k: int = 5) -> List[dict]:
        """
        Performs hybrid search:
        1. Query Expansion
        2. Vector Search (Semantic)
        3. Keyword Search (BM25) - acting as a filter or booster not fully implemented here for simplicity
        4. Combine results
        """
        # 1. Expand Queries (Optional: use these for vector search)
        queries = generate_multi_queries(job_description)
        print(f"Generated queries: {queries}")
        
        # 2. Vector Search (using the original JD and expanded queries)
        # Simple approach: Search with original JD
        vectorstore = get_vectorstore()
        results = vectorstore.similarity_search_with_score(job_description, k=k*2)
        
        # Convert to standardized format
        candidates = []
        for doc, score in results:
            candidates.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "vector_score": score
            })
            
        return candidates

# Instantiate singleton
search_pipeline = SearchPipeline()
