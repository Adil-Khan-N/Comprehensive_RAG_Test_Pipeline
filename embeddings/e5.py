"""E5 embeddings implementation using LangChain."""

from langchain_community.embeddings import HuggingFaceEmbeddings
from .base import BaseEmbeddings
from typing import List


class E5Embeddings(BaseEmbeddings):
    """E5 embeddings using LangChain HuggingFaceEmbeddings."""
    
    def __init__(self, model_name: str = "intfloat/e5-large-v2"):
        super().__init__()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs with E5 prefix."""
        # E5 models work better with prefixes
        prefixed_texts = [f"passage: {text}" for text in texts]
        return self.embeddings.embed_documents(prefixed_texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text with E5 prefix."""
        # E5 models work better with query prefix
        prefixed_text = f"query: {text}"
        return self.embeddings.embed_query(prefixed_text)