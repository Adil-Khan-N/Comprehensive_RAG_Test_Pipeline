#!/usr/bin/env python3
"""
Test script for LangChain-based embeddings and vector database implementations
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_embeddings():
    """Test all embedding implementations."""
    
    print("=" * 60)
    print("TESTING LANGCHAIN-BASED EMBEDDINGS")
    print("=" * 60)
    
    # Test Sentence Transformers (most reliable)
    print("\n1. Testing SentenceTransformersEmbeddings:")
    print("-" * 50)
    try:
        from embeddings import SentenceTransformersEmbeddings
        embeddings = SentenceTransformersEmbeddings()
        
        # Test single query
        query_result = embeddings.embed_query("This is a test query")
        print(f"✓ Query embedding created: {len(query_result)} dimensions")
        
        # Test document batch
        docs_result = embeddings.embed_documents(["Doc 1", "Doc 2"])
        print(f"✓ Document embeddings created: {len(docs_result)} docs")
        print(f"✓ Each document embedding: {len(docs_result[0])} dimensions")
        
    except Exception as e:
        print(f"✗ SentenceTransformersEmbeddings failed: {e}")
    
    # Test BGE embeddings
    print("\n2. Testing BGEEmbeddings:")
    print("-" * 50)
    try:
        from embeddings import BGEEmbeddings
        embeddings = BGEEmbeddings()
        
        query_result = embeddings.embed_query("This is a test query") 
        print(f"✓ BGE query embedding created: {len(query_result)} dimensions")
        
    except Exception as e:
        print(f"✗ BGEEmbeddings failed: {e}")


def test_chunking():
    """Test LangChain-based chunking."""
    
    print("\n" + "=" * 60)
    print("TESTING LANGCHAIN-BASED CHUNKING")
    print("=" * 60)
    
    sample_text = """# Introduction
    
This is a sample document for testing chunking strategies. It contains multiple paragraphs and sections to demonstrate different chunking approaches.

## Section 1: Technical Details

The RAG framework provides multiple chunking strategies:
- Fixed size chunking with overlap
- Semantic chunking based on similarity
- Structure-aware chunking preserving headers
- Hybrid approaches combining multiple strategies

## Section 2: Implementation

Here are the key implementation details that every developer should know.
This section contains technical information about the system architecture.
The implementation focuses on modularity and extensibility.
"""

    # Test Fixed Size Chunker
    print("\n1. Testing FixedSizeChunker:")
    print("-" * 40)
    try:
        from chunking import FixedSizeChunker
        chunker = FixedSizeChunker(chunk_size=200, overlap=20)
        chunks = chunker.chunk(sample_text)
        print(f"✓ Fixed chunker created {len(chunks)} chunks")
        print(f"✓ First chunk length: {len(chunks[0])}")
        
    except Exception as e:
        print(f"✗ FixedSizeChunker failed: {e}")

    # Test Structure Aware Chunker  
    print("\n2. Testing StructureAwareChunker:")
    print("-" * 40)
    try:
        from chunking import StructureAwareChunker
        chunker = StructureAwareChunker(preserve_headers=True)
        chunks = chunker.chunk(sample_text)
        print(f"✓ Structure-aware chunker created {len(chunks)} chunks")
        
    except Exception as e:
        print(f"✗ StructureAwareChunker failed: {e}")


def test_vector_stores():
    """Test LangChain-based vector stores."""
    
    print("\n" + "=" * 60)
    print("TESTING LANGCHAIN-BASED VECTOR STORES")
    print("=" * 60)
    
    # Initialize embeddings for vector stores
    try:
        from embeddings import SentenceTransformersEmbeddings
        embeddings = SentenceTransformersEmbeddings()
        print("✓ Embeddings initialized for vector store testing")
    except Exception as e:
        print(f"✗ Failed to initialize embeddings: {e}")
        return
    
    # Test FAISS Vector Store
    print("\n1. Testing FAISSVectorDB:")
    print("-" * 40)
    try:
        from vectordb import FAISSVectorDB
        
        vector_db = FAISSVectorDB(embeddings)
        
        # Add some texts
        texts = ["This is document 1", "This is document 2", "This is document 3"]
        ids = vector_db.add_texts(texts)
        print(f"✓ Added {len(ids)} texts to FAISS")
        
        # Search
        results = vector_db.similarity_search("document", k=2)
        print(f"✓ Search returned {len(results)} results")
        
    except Exception as e:
        print(f"✗ FAISSVectorDB failed: {e}")
    
    # Test ChromaDB Vector Store
    print("\n2. Testing ChromaVectorDB:")
    print("-" * 40)
    try:
        from vectordb import ChromaVectorDB
        
        vector_db = ChromaVectorDB(embeddings, collection_name="test_collection")
        
        # Add some texts
        texts = ["This is document A", "This is document B"]
        ids = vector_db.add_texts(texts)
        print(f"✓ Added {len(ids)} texts to ChromaDB")
        
        # Search
        results = vector_db.similarity_search("document", k=2)
        print(f"✓ Search returned {len(results)} results")
        
    except Exception as e:
        print(f"✗ ChromaVectorDB failed: {e}")


def main():
    """Run all tests."""
    
    print("🚀 Testing LangChain-based RAG Framework Components")
    
    try:
        test_embeddings()
    except Exception as e:
        print(f"❌ Embeddings tests failed: {e}")
    
    try:
        test_chunking()
    except Exception as e:
        print(f"❌ Chunking tests failed: {e}")
    
    try:
        test_vector_stores()
    except Exception as e:
        print(f"❌ Vector store tests failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 LANGCHAIN INTEGRATION TESTING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()