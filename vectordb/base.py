"""Base class for vector databases using LangChain."""

from langchain_core.vectorstores import VectorStore
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document


class BaseVectorDB:
    """Base class wrapping LangChain VectorStore interface."""
    
    def __init__(self):
        self.vectorstore: VectorStore = None
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> List[str]:
        """Add texts to the vector store.
        
        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dicts
            ids: Optional list of ids
            
        Returns:
            List of ids of the added texts
        """
        if self.vectorstore is None:
            raise NotImplementedError("vectorstore must be initialized in subclass")
        return self.vectorstore.add_texts(texts, metadatas=metadatas, ids=ids)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if self.vectorstore is None:
            raise NotImplementedError("vectorstore must be initialized in subclass")
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Search for similar documents with similarity scores.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if self.vectorstore is None:
            raise NotImplementedError("vectorstore must be initialized in subclass")
        return self.vectorstore.similarity_search_with_score(query, k=k)