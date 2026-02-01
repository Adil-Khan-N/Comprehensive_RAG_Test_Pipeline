"""Dense retrieval implementation."""

from typing import List, Dict, Any
import numpy as np


class DenseRetriever:
    """Dense retrieval using vector similarity."""
    
    def __init__(self, embeddings_model, vector_db):
        self.embeddings_model = embeddings_model
        self.vector_db = vector_db
        
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents using dense vector similarity.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with scores
        """
        query_vector = self.embeddings_model.embed_text(query)
        results = self.vector_db.search(query_vector, k=k)
        return results