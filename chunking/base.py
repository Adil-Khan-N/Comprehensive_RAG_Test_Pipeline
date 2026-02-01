"""Base class for text chunking strategies."""

from abc import ABC, abstractmethod
from typing import List


class BaseChunker(ABC):
    """Abstract base class for text chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Chunk text into smaller pieces.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        pass