#!/usr/bin/env python3
"""
Simple test script to check basic component imports and functionality.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_basic_imports():
    """Test basic imports to identify issues."""
    print("🔍 Testing Basic Imports...")
    
    # Test LangChain imports
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        print("✓ langchain_text_splitters.RecursiveCharacterTextSplitter")
    except ImportError as e:
        print(f"❌ langchain_text_splitters.RecursiveCharacterTextSplitter: {e}")
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("✓ langchain_community.embeddings.HuggingFaceEmbeddings")
    except ImportError as e:
        print(f"❌ langchain_community.embeddings.HuggingFaceEmbeddings: {e}")
    
    try:
        from langchain_community.vectorstores import FAISS
        print("✓ langchain_community.vectorstores.FAISS")
    except ImportError as e:
        print(f"❌ langchain_community.vectorstores.FAISS: {e}")
    
    try:
        from langchain_community.vectorstores import Chroma
        print("✓ langchain_community.vectorstores.Chroma")
    except ImportError as e:
        print(f"❌ langchain_community.vectorstores.Chroma: {e}")
    
    # Test sentence transformers
    try:
        import sentence_transformers
        print("✓ sentence_transformers")
    except ImportError as e:
        print(f"❌ sentence_transformers: {e}")
    
    # Test faiss
    try:
        import faiss
        print("✓ faiss")
    except ImportError as e:
        print(f"❌ faiss: {e}")

def test_custom_components():
    """Test our custom component imports."""
    print("\\n🔧 Testing Custom Components...")
    
    # Test chunking
    try:
        from chunking.fixed import FixedSizeChunker
        chunker = FixedSizeChunker(chunk_size=100)
        test_text = "This is a test text. " * 20
        chunks = chunker.chunk(test_text)
        print(f"✓ FixedSizeChunker: Created {len(chunks)} chunks")
    except Exception as e:
        print(f"❌ FixedSizeChunker: {e}")
    
    # Test embeddings
    try:
        from embeddings.sentence_transformers import SentenceTransformersEmbeddings
        embedding = SentenceTransformersEmbeddings()
        print("✓ SentenceTransformersEmbeddings: Created successfully")
    except Exception as e:
        print(f"❌ SentenceTransformersEmbeddings: {e}")
    
    # Test vector database
    try:
        from vectordb.faiss import FAISSVectorDB
        vectordb = FAISSVectorDB(dimension=384)
        print("✓ FAISSVectorDB: Created successfully")
    except Exception as e:
        print(f"❌ FAISSVectorDB: {e}")

def test_minimal_pipeline():
    """Test a minimal RAG pipeline."""
    print("\\n🚀 Testing Minimal RAG Pipeline...")
    
    try:
        # Simple chunking test
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        text = "This is a sample document. " * 50
        chunks = splitter.split_text(text)
        print(f"✓ Text splitting: {len(chunks)} chunks created")
        
        # Simple embedding test
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        sample_embedding = embeddings.embed_query("test query")
        print(f"✓ Embeddings: Dimension {len(sample_embedding)}")
        
        # Simple vector store test
        from langchain_community.vectorstores import FAISS
        vectorstore = FAISS.from_texts(chunks[:5], embeddings)  # Just first 5 chunks
        results = vectorstore.similarity_search("sample", k=2)
        print(f"✓ Vector search: Found {len(results)} similar documents")
        
        print("\\n🎉 Minimal pipeline test SUCCESSFUL!")
        return True
        
    except Exception as e:
        print(f"❌ Minimal pipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("🧪 RAG FRAMEWORK - BASIC FUNCTIONALITY TEST")
    print("="*60)
    
    test_basic_imports()
    test_custom_components()
    success = test_minimal_pipeline()
    
    print("\\n" + "="*60)
    if success:
        print("✅ BASIC TESTING COMPLETED SUCCESSFULLY")
        print("You can now try running the full test suite!")
    else:
        print("❌ BASIC TESTING FAILED")
        print("Please fix the issues above before running full tests.")
    print("="*60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())