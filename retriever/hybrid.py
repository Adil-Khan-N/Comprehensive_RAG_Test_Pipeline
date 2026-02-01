"""Hybrid retrieval combining dense and sparse methods."""

from typing import List, Dict, Any
from .dense import DenseRetriever
from .bm25 import BM25Retriever


class HybridRetriever:
    """Hybrid retrieval combining dense and sparse methods."""
    
    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever, 
                 dense_weight: float = 0.7, sparse_weight: float = 0.3):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents using hybrid approach.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with combined scores
        """
        # Get results from both retrievers
        dense_results = self.dense_retriever.retrieve(query, k=k*2)
        sparse_results = self.bm25_retriever.retrieve(query, k=k*2)
        
        # Combine scores
        combined_scores = {}
        
        # Process dense results
        for result in dense_results:
            doc_id = result["id"]
            combined_scores[doc_id] = {
                "dense_score": result["score"] * self.dense_weight,
                "sparse_score": 0,
                "metadata": result.get("metadata", {})
            }
            
        # Process sparse results
        for result in sparse_results:
            doc_id = result["id"]
            if doc_id in combined_scores:
                combined_scores[doc_id]["sparse_score"] = result["score"] * self.sparse_weight
            else:
                combined_scores[doc_id] = {
                    "dense_score": 0,
                    "sparse_score": result["score"] * self.sparse_weight,
                    "metadata": result.get("metadata", {})
                }
                
        # Calculate final scores and sort
        final_results = []
        for doc_id, scores in combined_scores.items():
            final_score = scores["dense_score"] + scores["sparse_score"]
            final_results.append({
                "id": doc_id,
                "score": final_score,
                "metadata": scores["metadata"]
            })
            
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:k]