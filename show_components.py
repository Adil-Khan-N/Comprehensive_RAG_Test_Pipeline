#!/usr/bin/env python3
"""
RAG FRAMEWORK COMPONENT INVENTORY
=================================

This script shows all available components and testing options.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def show_component_inventory():
    """Display all available RAG components."""
    
    print("="*80)
    print("📋 RAG FRAMEWORK - COMPONENT INVENTORY")
    print("="*80)
    
    chunking_strategies = [
        'FixedSizeChunker',
        'SemanticChunker', 
        'StructureAwareChunker',
        'HybridChunker'
    ]
    
    embedding_models = [
        'SentenceTransformersEmbeddings',
        'BGEEmbeddings',
        'E5Embeddings', 
        'SciBERTEmbeddings',
        'LayoutLMEmbeddings',
        'OpenAIEmbeddings'
    ]
    
    vector_databases = [
        'FAISSVectorDB',
        'ChromaVectorDB',
        'MilvusVectorDB',
        'QdrantVectorDB', 
        'WeaviateVectorDB',
        'PineconeVectorDB'
    ]
    
    print("🔧 AVAILABLE COMPONENTS:")
    print()
    
    print("1️⃣  CHUNKING STRATEGIES (4 total):")
    for i, strategy in enumerate(chunking_strategies, 1):
        print(f"   {i}. {strategy}")
    
    print()
    print("2️⃣  EMBEDDING MODELS (6 total):")
    for i, model in enumerate(embedding_models, 1):
        print(f"   {i}. {model}")
    
    print()
    print("3️⃣  VECTOR DATABASES (6 total):")
    for i, db in enumerate(vector_databases, 1):
        print(f"   {i}. {db}")
    
    print()
    print("📊 TESTING COMBINATIONS:")
    total_combinations = len(chunking_strategies) * len(embedding_models) * len(vector_databases)
    print(f"   • Full Testing: {len(chunking_strategies)} × {len(embedding_models)} × {len(vector_databases)} = {total_combinations} pipeline combinations")
    print(f"   • Each combination tested with 10 technical questions")
    print(f"   • Total individual tests: {total_combinations * 10}")
    
    print()
    print("⚡ TESTING OPTIONS:")
    print()
    print("1. QUICK TEST (Recommended for first try):")
    print(f"   • Command: python test_quick_sample.py")
    print(f"   • Combinations: 12 representative samples")
    print(f"   • Runtime: ~8-15 minutes")
    print(f"   • Tests: 120 individual tests")
    
    print()
    print("2. FULL COMPREHENSIVE TEST:")
    print(f"   • Command: python test_all_combinations.py")
    print(f"   • Combinations: {total_combinations} complete matrix")
    print(f"   • Runtime: ~45-90 minutes") 
    print(f"   • Tests: {total_combinations * 10} individual tests")
    
    print()
    print("3. BATCH EXECUTION:")
    print(f"   • Command: run_all_tests.bat")
    print(f"   • Same as option 2, with Windows batch automation")
    
    print("="*80)

def main():
    """Main function."""
    show_component_inventory()
    return 0

if __name__ == "__main__":
    sys.exit(main())