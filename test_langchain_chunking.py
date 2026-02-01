#!/usr/bin/env python3
"""
Test script for LangChain-based chunking implementations
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from chunking import FixedSizeChunker, SemanticChunker, StructureAwareChunker, HybridChunker

def test_chunking_implementations():
    """Test all chunking implementations with sample text."""
    
    # Sample text for testing
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

## Section 3: Performance Considerations  

When choosing a chunking strategy, consider the following factors:
1. Document structure and complexity
2. Target chunk size requirements
3. Processing time constraints
4. Memory usage limitations

The system should handle various document types efficiently while maintaining good retrieval performance.
"""

    print("=" * 60)
    print("TESTING LANGCHAIN-BASED CHUNKING IMPLEMENTATIONS")
    print("=" * 60)
    
    # Test Fixed Size Chunker
    print("\n1. Testing FixedSizeChunker:")
    print("-" * 40)
    try:
        fixed_chunker = FixedSizeChunker(chunk_size=200, overlap=20)
        fixed_chunks = fixed_chunker.chunk(sample_text)
        print(f"✓ Fixed chunker created successfully")
        print(f"✓ Generated {len(fixed_chunks)} chunks")
        print(f"✓ Sample chunk: '{fixed_chunks[0][:80]}...'")
    except Exception as e:
        print(f"✗ Fixed chunker failed: {e}")
    
    # Test Semantic Chunker  
    print("\n2. Testing SemanticChunker:")
    print("-" * 40)
    try:
        semantic_chunker = SemanticChunker(similarity_threshold=0.7)
        semantic_chunks = semantic_chunker.chunk(sample_text)
        print(f"✓ Semantic chunker created successfully")
        print(f"✓ Generated {len(semantic_chunks)} chunks")
        print(f"✓ Sample chunk: '{semantic_chunks[0][:80]}...'")
    except Exception as e:
        print(f"✗ Semantic chunker failed: {e}")
    
    # Test Structure Aware Chunker
    print("\n3. Testing StructureAwareChunker:")
    print("-" * 40)
    try:
        structure_chunker = StructureAwareChunker(preserve_headers=True)
        structure_chunks = structure_chunker.chunk(sample_text)
        print(f"✓ Structure-aware chunker created successfully")
        print(f"✓ Generated {len(structure_chunks)} chunks")
        print(f"✓ Sample chunk: '{structure_chunks[0][:80]}...'")
    except Exception as e:
        print(f"✗ Structure-aware chunker failed: {e}")
    
    # Test Hybrid Chunker
    print("\n4. Testing HybridChunker:")
    print("-" * 40)
    try:
        hybrid_chunker = HybridChunker(use_semantic=True, use_fixed=True)
        hybrid_chunks = hybrid_chunker.chunk(sample_text)
        print(f"✓ Hybrid chunker created successfully") 
        print(f"✓ Generated {len(hybrid_chunks)} chunks")
        print(f"✓ Sample chunk: '{hybrid_chunks[0][:80]}...'")
    except Exception as e:
        print(f"✗ Hybrid chunker failed: {e}")
    
    print("\n" + "=" * 60)
    print("CHUNKING TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    test_chunking_implementations()