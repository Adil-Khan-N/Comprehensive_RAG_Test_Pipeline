"""Milvus vector database implementation using LangChain."""

from langchain_milvus import Milvus
from .base import BaseVectorDB
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document


class MilvusVectorDB(BaseVectorDB):
    """Milvus vector database using LangChain implementation."""
    
    def __init__(self, embeddings, collection_name: str = "rag_collection", 
                 connection_args: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.embeddings = embeddings
        self.collection_name = collection_name
        
        # Default connection args for local Milvus
        if connection_args is None:
            connection_args = {"host": "127.0.0.1", "port": "19530"}
            
        self.vectorstore = Milvus(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args=connection_args
        )
        
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> List[str]:
        """Add texts to Milvus vector store."""
        return self.vectorstore.add_texts(texts, metadatas=metadatas, ids=ids)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents in Milvus."""
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Search with similarity scores."""
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def drop(self):
        """Drop the Milvus collection."""
        return self.vectorstore.drop()
        # TODO: Implement Milvus vector insertion
        for i, (vector, id_) in enumerate(zip(vectors, ids)):
            self.vectors_map[id_] = vector
            if metadata:
                self.metadata_map[id_] = metadata[i]
        
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search using Milvus."""
        # TODO: Implement Milvus search
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