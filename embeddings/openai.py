"""OpenAI embeddings implementation using LangChain."""

from langchain_openai import OpenAIEmbeddings as LangChainOpenAIEmbeddings
from .base import BaseEmbeddings
from typing import List


class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI embeddings using LangChain implementation."""
    
    def __init__(self, model: str = "text-embedding-ada-002", openai_api_key: str = None):
        super().__init__()
        self.embeddings = LangChainOpenAIEmbeddings(
            model=model,
            openai_api_key=openai_api_key
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embeddings.embed_query(text)