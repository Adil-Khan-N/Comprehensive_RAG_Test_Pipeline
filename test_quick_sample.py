#!/usr/bin/env python3
"""
QUICK RAG PIPELINE TESTING - Representative Sample
==================================================

This script tests a representative sample of pipeline combinations for quick validation:
- 2 Chunking strategies (Fixed, Semantic)
- 3 Embedding models (SentenceTransformers, BGE, E5)  
- 2 Vector databases (FAISS, ChromaDB)

Total combinations: 2 × 3 × 2 = 12 pipeline configurations
Each tested against 10 technical questions = 120 total tests

Usage: python test_quick_sample.py
Estimated time: 8-15 minutes
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def check_prerequisites():
    """Check if required packages are installed."""
    print("🔍 Checking prerequisites for quick test...")
    
    # Check minimal required packages for quick test
    package_checks = [
        ('langchain', 'langchain'),
        ('langchain_community', 'langchain_community'), 
        ('sentence_transformers', 'sentence_transformers'),
        ('chromadb', 'chromadb'),
        ('faiss', 'faiss'),  # faiss-cpu package imports as 'faiss'
        ('pandas', 'pandas'),
        ('google.generativeai', 'google.generativeai')
    ]
    
    missing_packages = []
    for display_name, import_name in package_checks:
        try:
            __import__(import_name)
            print(f"✓ {display_name} available")
        except ImportError:
            missing_packages.append(display_name)
            print(f"❌ {display_name} missing")
    
    if missing_packages:
        print(f"\\n❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 QUICK INSTALLATION:")
        print("   pip install -r requirements_minimal.txt")
        print("\\n📦 OR install manually:")
        print("   pip install langchain langchain-community sentence-transformers")
        print("   pip install faiss-cpu chromadb google-generativeai pandas")
        return False
    
    print("✓ All prerequisites met!")
    return True

def main():
    """Run quick sample testing."""
    
    print("="*80)
    print("⚡ QUICK RAG PIPELINE TESTING - REPRESENTATIVE SAMPLE")
    print("="*80)
    print("📊 Quick Testing Matrix:")
    print("   • 2 Chunking Strategies: Fixed, Semantic")
    print("   • 3 Embedding Models: SentenceTransformers, BGE, E5") 
    print("   • 2 Vector Databases: FAISS, ChromaDB")
    print("   • 10 Technical Questions per combination")
    print()
    print(f"📈 Total Pipeline Combinations: 12 (2 × 3 × 2)")
    print(f"📈 Total Tests: 120 individual tests")
    print(f"⏱️  Estimated Runtime: 8-15 minutes")
    print("="*80)
    print()
    
    # Check prerequisites first
    if not check_prerequisites():
        return 1
    
    print("\\nThis quick test will give you a good sense of relative performance")
    print("without running all 144 combinations (which takes 45-90 minutes).")
    print()
    
    response = input("Continue with quick test? (y/N): ").strip().lower()
    if response != 'y':
        print("Quick test cancelled.")
        return 0
    
    try:
        # Import and modify the tester for quick mode
        from comprehensive_rag_test import ComprehensiveRAGTester
        
        # Create a modified tester with fewer combinations
        tester = ComprehensiveRAGTester()
        
        # Override combinations for quick test
        tester.chunking_strategies = ['FixedSizeChunker', 'SemanticChunker']
        tester.embedding_models = ['SentenceTransformersEmbeddings', 'BGEEmbeddings', 'E5Embeddings']
        tester.vector_databases = ['FAISSVectorDB', 'ChromaVectorDB']
        
        # Regenerate combinations
        tester.pipeline_combinations = tester._generate_all_combinations()
        
        print(f"\\n🚀 Starting quick test with {len(tester.pipeline_combinations)} combinations...")
        
        # Create output directory
        os.makedirs('output', exist_ok=True)
        
        # Run test
        results_df = tester.run_comprehensive_test()
        
        if results_df.empty:
            print("❌ No results generated!")
            return 1
        
        # Generate report
        summary_df = tester.generate_comprehensive_report(results_df)
        
        print("\\n🎉 QUICK TEST COMPLETED!")
        print(f"📊 Tests Completed: {len(results_df)}")
        print(f"📊 Success Rate: {results_df['success'].mean()*100:.1f}%")
        
        # Show top 5 results
        if len(summary_df) > 0:
            print("\\n🏆 TOP 5 QUICK TEST RESULTS:")
            print("-"*60)
            top_5 = summary_df.head(5)
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                print(f"{i}. {row['Avg Score']} | {row['Pipeline']}")
        
        print("\\n✅ Quick test complete! For full testing, use: python test_all_combinations.py")
        return 0
        
    except Exception as e:
        print(f"❌ Error during quick testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())