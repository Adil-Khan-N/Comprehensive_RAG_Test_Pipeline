"""Structure-aware text chunking implementation."""

from typing import List
from .base import BaseChunker
import re


class StructureAwareChunker(BaseChunker):
    """Chunks text while preserving document structure."""
    
    def __init__(self, preserve_headers: bool = True):
        self.preserve_headers = preserve_headers
        
    def chunk(self, text: str) -> List[str]:
        """Chunk text while preserving structure like headers and paragraphs."""
        chunks = []
        
        # Split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for paragraph in paragraphs:
            # Check if paragraph is a header (starts with #)
            is_header = paragraph.strip().startswith('#')
            
            if len(current_chunk + paragraph) > 800 and not is_header:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks