"""Base class for text embeddings."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEmbeddings(ABC):
    """Abstract base class for text embedding models."""
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embeddings for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array of embeddings
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of numpy arrays containing embeddings
        """
        pass