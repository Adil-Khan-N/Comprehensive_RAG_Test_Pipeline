"""Semantic-based text chunking implementation using LangChain."""

from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from .base import BaseChunker


class SemanticChunker(BaseChunker):
    """Chunks text based on semantic similarity using LangChain SemanticChunker."""
    
    def __init__(self, similarity_threshold: float = 0.8, embeddings=None):
        super().__init__()
        self.similarity_threshold = similarity_threshold
        
        # Use default embeddings if none provided
        if embeddings is None:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        self.text_splitter = LangChainSemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=similarity_threshold * 100
        )