"""Base class for vector databases."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class BaseVectorDB(ABC):
    """Abstract base class for vector databases."""
    
    @abstractmethod
    def add_vectors(self, vectors: List[np.ndarray], ids: List[str], 
                   metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add vectors to the database.
        
        Args:
            vectors: List of embedding vectors
            ids: List of unique identifiers
            metadata: Optional metadata for each vector
        """
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of search results with ids, scores, and metadata
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete vectors by ids."""
        pass