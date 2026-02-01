"""Base class for text chunking strategies using LangChain."""

from abc import ABC, abstractmethod
from typing import List
from langchain_text_splitters import TextSplitter


class BaseChunker(ABC):
    """Abstract base class for text chunking strategies."""
    
    def __init__(self):
        self.text_splitter: TextSplitter = None
    
    def chunk(self, text: str) -> List[str]:
        """Chunk text into smaller pieces using LangChain text splitter.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if self.text_splitter is None:
            raise NotImplementedError("text_splitter must be initialized in subclass")
        
        return self.text_splitter.split_text(text)