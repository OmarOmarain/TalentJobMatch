from rank_bm25 import BM25Okapi
from typing import List, Tuple

class BM25Index:
    def __init__(self, corpus: List[str]):
        """
        Initialize BM25 with a list of text documents.
        args:
            corpus: List of strings (document contents)
        """
        self.corpus = corpus
        # Simple tokenization by splitting on whitespace
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search the corpus for the query.
        Returns:
            List of tuples (document_content, score)
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Zip scores with documents and sort
        doc_scores = list(zip(self.corpus, scores))
        sorted_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        
        return sorted_docs[:k]
