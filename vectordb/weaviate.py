"""Weaviate vector database implementation using LangChain."""

from langchain_weaviate.vectorstores import WeaviateVectorStore
from .base import BaseVectorDB
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import weaviate


class WeaviateVectorDB(BaseVectorDB):
    """Weaviate vector database using LangChain implementation."""
    
    def __init__(self, embeddings, url: str = "http://localhost:8080", 
                 index_name: str = "RAGCollection", auth_config=None):
        super().__init__()
        self.embeddings = embeddings
        self.index_name = index_name
        
        # Initialize Weaviate client
        if auth_config:
            client = weaviate.Client(url=url, auth_client_secret=auth_config)
        else:
            client = weaviate.Client(url=url)
            
        self.vectorstore = WeaviateVectorStore(
            client=client,
            index_name=index_name,
            text_key="text",
            embedding=embeddings
        )
        
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> List[str]:
        """Add texts to Weaviate vector store."""
        return self.vectorstore.add_texts(texts, metadatas=metadatas, ids=ids)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents in Weaviate."""
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Search with similarity scores."""
        return self.vectorstore.similarity_search_with_score(query, k=k)
            if metadata:
                self.metadata_map[id_] = metadata[i]
        
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search using Weaviate."""
        # TODO: Implement Weaviate vector search
        results = []
        for i, (id_, vector) in enumerate(list(self.vectors_map.items())[:k]):
            score = np.dot(query_vector, vector)
            results.append({
                "id": id_,
                "score": float(score),
                "metadata": self.metadata_map.get(id_, {})
            })
        return results
        
    def delete(self, ids: List[str]) -> None:
        """Delete vectors by ids."""
        for id_ in ids:
            self.vectors_map.pop(id_, None)
            self.metadata_map.pop(id_, None)