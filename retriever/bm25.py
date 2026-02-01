"""BM25 sparse retrieval implementation."""

from typing import List, Dict, Any
import math
from collections import Counter


class BM25Retriever:
    """BM25 sparse retrieval implementation."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_freqs = []
        self.idf = {}
        self.avgdl = 0
        
    def fit(self, documents: List[str]) -> None:
        """Fit BM25 on document corpus.
        
        Args:
            documents: List of documents to index
        """
        self.documents = documents
        self.doc_freqs = []
        
        # Calculate document frequencies and IDF
        df = {}
        for doc in documents:
            terms = doc.lower().split()
            doc_freq = Counter(terms)
            self.doc_freqs.append(doc_freq)
            
            for term in set(terms):
                df[term] = df.get(term, 0) + 1
                
        # Calculate IDF scores
        N = len(documents)
        for term, freq in df.items():
            self.idf[term] = math.log((N - freq + 0.5) / (freq + 0.5))
            
        # Calculate average document length
        self.avgdl = sum(len(doc.split()) for doc in documents) / len(documents)
        
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents using BM25 scoring.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with BM25 scores
        """
        query_terms = query.lower().split()
        scores = []
        
        for i, doc_freq in enumerate(self.doc_freqs):
            score = 0
            doc_len = sum(doc_freq.values())
            
            for term in query_terms:
                if term in doc_freq:
                    tf = doc_freq[term]
                    idf = self.idf.get(term, 0)
                    
                    # BM25 formula
                    score += idf * (tf * (self.k1 + 1)) / (
                        tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    )
                    
            scores.append({"id": str(i), "score": score, "text": self.documents[i]})
            
        # Sort by score and return top k
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:k]