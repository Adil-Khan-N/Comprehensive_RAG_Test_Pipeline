"""FAISS vector database implementation using LangChain."""

from langchain_community.vectorstores import FAISS
from .base import BaseVectorDB
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document


class FAISSVectorDB(BaseVectorDB):
    """FAISS vector database using LangChain implementation."""
    
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings
        self.vectorstore = None
        
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> List[str]:
        """Add texts to FAISS vector store."""
        if self.vectorstore is None:
            # Initialize FAISS with first batch of texts
            self.vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas, ids=ids)
            return ids or [str(i) for i in range(len(texts))]
        else:
            # Add to existing FAISS index
            return self.vectorstore.add_texts(texts, metadatas=metadatas, ids=ids)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents in FAISS."""
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Search with similarity scores."""
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def save_local(self, folder_path: str):
        """Save FAISS index locally."""
        if self.vectorstore is not None:
            self.vectorstore.save_local(folder_path)
    
    @classmethod
    def load_local(cls, folder_path: str, embeddings):
        """Load FAISS index from local storage."""
        instance = cls(embeddings)
        instance.vectorstore = FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)
        return instance
        # TODO: Implement FAISS search
        # For now, return mock results
        results = []
        for i, (id_, vector) in enumerate(list(self.vectors_map.items())[:k]):
            score = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
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