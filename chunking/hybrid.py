"""Hybrid text chunking combining multiple strategies."""

from typing import List
from .base import BaseChunker
from .fixed import FixedSizeChunker
from .semantic import SemanticChunker


class HybridChunker(BaseChunker):
    """Combines multiple chunking strategies."""
    
    def __init__(self, use_semantic: bool = True, use_fixed: bool = True):
        self.use_semantic = use_semantic
        self.use_fixed = use_fixed
        self.semantic_chunker = SemanticChunker() if use_semantic else None
        self.fixed_chunker = FixedSizeChunker() if use_fixed else None
        
    def chunk(self, text: str) -> List[str]:
        """Apply hybrid chunking strategy."""
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