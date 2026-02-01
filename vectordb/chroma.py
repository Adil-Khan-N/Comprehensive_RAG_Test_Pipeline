"""ChromaDB vector database implementation using LangChain."""

from langchain_community.vectorstores import Chroma
from .base import BaseVectorDB
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import tempfile


class ChromaVectorDB(BaseVectorDB):
    """ChromaDB vector database using LangChain implementation."""
    
    def __init__(self, embeddings, collection_name: str = "rag_collection", persist_directory: Optional[str] = None):
        super().__init__()
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.persist_directory = persist_directory or tempfile.mkdtemp()
        self.vectorstore = None
        
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> List[str]:
        """Add texts to ChromaDB vector store."""
        if self.vectorstore is None:
            # Initialize ChromaDB with first batch of texts
            self.vectorstore = Chroma.from_texts(
                texts, 
                self.embeddings, 
                metadatas=metadatas, 
                ids=ids,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
            return ids or [str(i) for i in range(len(texts))]
        else:
            # Add to existing ChromaDB collection
            return self.vectorstore.add_texts(texts, metadatas=metadatas, ids=ids)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents in ChromaDB."""
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Search with similarity scores."""
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def persist(self):
        """Persist the ChromaDB collection."""
        if self.vectorstore is not None:
            self.vectorstore.persist()
        
    def delete(self, ids: List[str]) -> None:
        """Delete vectors by ids."""
        for id_ in ids:
            self.vectors_map.pop(id_, None)
            self.metadata_map.pop(id_, None)