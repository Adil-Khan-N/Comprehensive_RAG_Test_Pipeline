"""SciBERT embeddings implementation using LangChain."""

from langchain_community.embeddings import HuggingFaceEmbeddings
from .base import BaseEmbeddings
from typing import List


class SciBERTEmbeddings(BaseEmbeddings):
    """SciBERT embeddings using LangChain HuggingFaceEmbeddings."""
    
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased"):
        super().__init__()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embeddings.embed_query(text)