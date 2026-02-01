#!/usr/bin/env python3
"""
Quick execution script for comprehensive RAG pipeline testing.
Run this to test all pipeline combinations with technical questions.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def main():
    """Run comprehensive RAG testing with proper error handling."""
    
    try:
        from comprehensive_rag_test import main as run_comprehensive_test
        
        print("🚀 Starting Comprehensive RAG Pipeline Testing...")
        print("📋 Testing 10 technical questions across 5 pipeline combinations")
        print("⏱️  Expected runtime: 5-10 minutes")
        print("="*60)
        
        # Run the comprehensive test
        run_comprehensive_test()
        
        print("\\n✅ Testing completed! Check the output directory for detailed reports.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed:")
        print("pip install -r requirements_gemini.txt")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Check the log files for detailed error information.")

if __name__ == "__main__":
    main()