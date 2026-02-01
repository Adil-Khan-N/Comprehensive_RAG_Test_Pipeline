"""LayoutLM embeddings implementation using LangChain."""

from langchain_community.embeddings import HuggingFaceEmbeddings
from .base import BaseEmbeddings
from typing import List


class LayoutLMEmbeddings(BaseEmbeddings):
    """LayoutLM embeddings using LangChain HuggingFaceEmbeddings."""
    
    def __init__(self, model_name: str = "microsoft/layoutlm-base-uncased"):
        super().__init__()
        # Note: LayoutLM requires special handling for layout info, 
        # this is a simplified text-only version
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs (text-only, layout info not used)."""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text (text-only, layout info not used)."""
        return self.embeddings.embed_query(text)