"""Base class for text embeddings using LangChain."""

from langchain_core.embeddings import Embeddings
from typing import List


class BaseEmbeddings(Embeddings):
    """Base class wrapping LangChain Embeddings interface."""
    
    def __init__(self):
        super().__init__()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs.
        
        Args:
            texts: List of text to embed.
            
        Returns:
            List of embeddings, one for each text.
        """
        raise NotImplementedError
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding for the text.
        """
        raise NotImplementedError

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        pass