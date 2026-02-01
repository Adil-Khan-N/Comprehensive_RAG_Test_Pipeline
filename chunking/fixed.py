"""Fixed-size text chunking implementation."""

from typing import List
from .base import BaseChunker


class FixedSizeChunker(BaseChunker):
    """Chunks text into fixed-size pieces."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk(self, text: str) -> List[str]:
        """Chunk text into fixed-size pieces with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap
            
        return chunks