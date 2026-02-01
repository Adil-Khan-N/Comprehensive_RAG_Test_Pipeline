"""Hybrid text chunking combining multiple LangChain strategies."""

from typing import List
from .base import BaseChunker
from .fixed import FixedSizeChunker
from .semantic import SemanticChunker


class HybridChunker(BaseChunker):
    """Combines multiple LangChain chunking strategies."""
    
    def __init__(self, use_semantic: bool = True, use_fixed: bool = True, 
                 semantic_threshold: float = 0.8, chunk_size: int = 512, chunk_overlap: int = 50):
        super().__init__()
        self.use_semantic = use_semantic
        self.use_fixed = use_fixed
        
        # Initialize chunkers with LangChain implementations
        self.semantic_chunker = SemanticChunker(similarity_threshold=semantic_threshold) if use_semantic else None
        self.fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_overlap) if use_fixed else None
        
    def chunk(self, text: str) -> List[str]:
        """Apply hybrid chunking strategy using LangChain components."""
        chunks = []
        
        if self.semantic_chunker:
            semantic_chunks = self.semantic_chunker.chunk(text)
            
            # Apply fixed chunking to large semantic chunks
            for chunk in semantic_chunks:
                if len(chunk) > 1000 and self.fixed_chunker:
                    sub_chunks = self.fixed_chunker.chunk(chunk)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(chunk)
        else:
            chunks = self.fixed_chunker.chunk(text)
            
        return chunks