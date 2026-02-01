"""Structure-aware text chunking implementation using LangChain."""

from typing import List
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from .base import BaseChunker


class StructureAwareChunker(BaseChunker):
    """Chunks text while preserving document structure using LangChain splitters."""
    
    def __init__(self, preserve_headers: bool = True, chunk_size: int = 800, chunk_overlap: int = 100):
        super().__init__()
        self.preserve_headers = preserve_headers
        
        # Define headers to split on
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        # Create markdown header text splitter
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=not preserve_headers
        )
        
        # Create a character splitter for further chunking if needed
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    
    def chunk(self, text: str) -> List[str]:
        """Chunk text while preserving structure like headers and paragraphs."""
        # First split by headers
        header_splits = self.markdown_splitter.split_text(text)
        
        # Then split large chunks further
        all_chunks = []
        for doc in header_splits:
            content = doc.page_content if hasattr(doc, 'page_content') else doc
            if len(content) > 800:
                sub_chunks = self.text_splitter.split_text(content)
                all_chunks.extend(sub_chunks)
            else:
                all_chunks.append(content)
                
        return all_chunks