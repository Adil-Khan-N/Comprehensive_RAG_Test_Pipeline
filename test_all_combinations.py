#!/usr/bin/env python3
"""
ONE-COMMAND RAG PIPELINE TESTING
================================

This script runs comprehensive testing of ALL possible combinations of:
- 4 Chunking strategies (Fixed, Semantic, Structure-Aware, Hybrid)
- 5 Embedding models (SentenceTransformers, BGE, E5, SciBERT, LayoutLM)  
- 2 Vector databases (FAISS, ChromaDB)

Total combinations: 4 × 5 × 2 = 40 pipeline configurations
Each tested against 10 technical questions = 400 total tests

Usage: python test_all_combinations.py
"""

import sys
import os
import time
from datetime import datetime
from pathlib import Path

def print_banner():
    """Print startup banner."""
    print("="*80)
    print("🚀 COMPREHENSIVE RAG PIPELINE TESTING - ALL COMBINATIONS")
    print("="*80)
    print("📊 Testing Matrix:")
    print("   • 4 Chunking Strategies: Fixed, Semantic, Structure-Aware, Hybrid")
    print("   • 6 Embedding Models: SentenceTransformers, BGE, E5, SciBERT, LayoutLM, OpenAI") 
    print("   • 6 Vector Databases: FAISS, ChromaDB, Milvus, Qdrant, Weaviate, Pinecone")
    print("   • 10 Technical Questions per combination")
    print()
    print(f"📈 Total Pipeline Combinations: 144 (4 × 6 × 6)")
    print(f"📈 Total Tests: 1,440 individual tests")
    print(f"⏱️  Estimated Runtime: 45-90 minutes")
    print("="*80)

def check_prerequisites():
    """Check if all required components are available."""
    print("🔍 Checking prerequisites...")
    
    # Check environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    gemini_key = os.getenv('GEMINI_API_KEY')
    if not gemini_key:
        print("❌ GEMINI_API_KEY not found in .env file!")
        print("   Please add your Gemini API key to .env file:")
        print("   GEMINI_API_KEY=your_key_here")
        return False
    
    print("✓ Gemini API key found")
    
    # Check required packages (with correct import names)
    package_checks = [
        ('langchain', 'langchain'),
        ('langchain_community', 'langchain_community'), 
        ('langchain_experimental', 'langchain_experimental'),  # For SemanticChunker
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
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 INSTALLATION OPTIONS:")
        print("   • Full system: pip install -r requirements.txt")
        print("   • Minimal (quick test): pip install -r requirements_minimal.txt") 
        print("   • Legacy: pip install -r requirements_gemini.txt")
        return False
    
    print("✓ All prerequisites met!")
    return True

def run_comprehensive_testing():
    """Run the comprehensive testing pipeline."""
    try:
        # Import and run the comprehensive test
        from comprehensive_rag_test import ComprehensiveRAGTester
        
        print("\n🏃 Starting comprehensive testing...")
        start_time = time.time()
        
        # Initialize tester
        tester = ComprehensiveRAGTester()
        
        print(f"📋 Generated {len(tester.pipeline_combinations)} pipeline combinations")
        
        # Create output directory
        os.makedirs('output', exist_ok=True)
        
        # Run comprehensive test
        results_df = tester.run_comprehensive_test()
        
        if results_df.empty:
            print("❌ No results generated!")
            return False
        
        # Generate comprehensive report
        print("\n📊 Generating comprehensive reports...")
        summary_df = tester.generate_comprehensive_report(results_df)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🎉 TESTING COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total Runtime: {duration/60:.1f} minutes")
        print(f"📊 Tests Completed: {len(results_df)}")
        print(f"📊 Successful Tests: {results_df['success'].sum()}")
        print(f"📊 Success Rate: {results_df['success'].mean()*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_top_performers():
    """Show top performing pipelines."""
    try:
        import pandas as pd
        import glob
        
        # Find the latest summary file
        summary_files = glob.glob("output/comprehensive_rag_test_summary_*.csv")
        if not summary_files:
            print("❌ No summary files found")
            return
        
        latest_file = max(summary_files, key=os.path.getctime)
        df = pd.read_csv(latest_file)
        
        print("\n🏆 TOP 5 PERFORMING PIPELINES:")
        print("="*60)
        
        # Extract numerical scores for sorting
        df['_score'] = df['Average Score'].str.extract('(\\d+\\.\\d+)').astype(float)
        top_5 = df.nlargest(5, '_score')
        
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"{i}. {row['Pipeline']}")
            print(f"   Score: {row['Average Score']} | Success: {row['Success Rate']}")
            print(f"   Config: {row['Description']}")
            print()
        
        print("📄 Check output/ directory for detailed reports!")
        
    except Exception as e:
        print(f"❌ Error showing results: {e}")

def main():
    """Main execution function."""
    print_banner()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix issues above and try again.")
        return 1
    
    # Confirm execution
    print("\n⚡ Ready to start comprehensive testing!")
    response = input("Continue? (y/N): ").strip().lower()
    if response != 'y':
        print("Testing cancelled.")
        return 0
    
    # Run testing
    success = run_comprehensive_testing()
    
    if success:
        show_top_performers()
        print("\n✅ ALL DONE! Check the output/ directory for detailed results.")
        return 0
    else:
        print("\n❌ Testing failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())