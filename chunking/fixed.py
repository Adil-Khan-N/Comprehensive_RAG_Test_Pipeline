"""Fixed-size text chunking implementation using LangChain."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base import BaseChunker


class FixedSizeChunker(BaseChunker):
    """Chunks text into fixed-size pieces using LangChain RecursiveCharacterTextSplitter."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            is_separator_regex=False
        )