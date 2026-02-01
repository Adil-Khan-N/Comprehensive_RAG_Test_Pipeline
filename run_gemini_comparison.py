"""Simple runner script for RAG comparison with Gemini Pro."""

import os
import sys
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_comparison import EnhancedRAGComparator
from config import GEMINI_API_KEY, TEST_CONFIGURATIONS, INDUSTRIAL_TEST_CONFIG


def validate_setup():
    """Validate that everything is set up correctly."""
    print("🔧 Validating RAG Framework Setup...")
    
    issues = []
    
    # Check API key
    if GEMINI_API_KEY == "your-gemini-pro-api-key-here":
        issues.append("❌ Gemini API key not set in config.py")
    else:
        print("✅ Gemini API key configured")
    
    # Check document file
    doc_path = INDUSTRIAL_TEST_CONFIG['document_path']
    if not os.path.exists(doc_path):
        issues.append(f"❌ Document not found: {doc_path}")
    else:
        print(f"✅ Document found: {doc_path}")
    
    # Check questions file
    questions_path = INDUSTRIAL_TEST_CONFIG['questions_path']
    if not os.path.exists(questions_path):
        issues.append(f"❌ Questions file not found: {questions_path}")
    else:
        print(f"✅ Questions file found: {questions_path}")
    
    # Check results directory
    results_dir = INDUSTRIAL_TEST_CONFIG['output_directory']
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"✅ Created results directory: {results_dir}")
    else:
        print(f"✅ Results directory ready: {results_dir}")
    
    if issues:
        print("\\n⚠️ Setup Issues Found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    
    print("\\n✅ All setup validation checks passed!")
    return True


def run_gemini_comparison():
    """Run the RAG comparison with Gemini Pro."""
    
    print("\\n" + "="*80)
    print("🚀 RAG PIPELINE COMPARISON WITH GEMINI PRO")
    print("="*80)
    
    print(f"📄 Document: {INDUSTRIAL_TEST_CONFIG['document_path']}")
    print(f"❓ Questions: {len(TEST_CONFIGURATIONS)} configurations × 30 questions = {len(TEST_CONFIGURATIONS) * 30} total tests")
    print(f"🤖 LLM: Google Gemini Pro")
    print(f"📊 Output: {INDUSTRIAL_TEST_CONFIG['output_directory']}")
    
    # Initialize comparator
    comparator = EnhancedRAGComparator(
        document_path=INDUSTRIAL_TEST_CONFIG['document_path'],
        questions_path=INDUSTRIAL_TEST_CONFIG['questions_path']
    )
    
    # Run comparison
    print(f"\\n⏰ Starting comparison at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    df, results = comparator.compare_configurations(TEST_CONFIGURATIONS)
    
    if len(df) == 0:
        print("❌ No configurations completed successfully")
        return
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{INDUSTRIAL_TEST_CONFIG['output_directory']}gemini_comparison_{timestamp}.json"
    report = comparator.generate_detailed_report(df, results, output_path)
    
    # Display results summary
    print_results_summary(report, df)


def print_results_summary(report, df):
    """Print a formatted summary of results."""
    
    print("\\n" + "="*80)
    print("🏆 GEMINI PRO RAG COMPARISON RESULTS")
    print("="*80)
    
    # Overall statistics
    print(f"📊 Configurations Tested: {len(df)}")
    print(f"📝 Questions per Config: 30")
    print(f"🎯 Total Evaluations: {len(df) * 30}")
    
    if len(df) > 0:
        print("\\n🥇 PERFORMANCE CHAMPIONS:")
        print("-" * 50)
        
        # Overall winner
        best = report['champions']['overall_best']
        print(f"🏆 Overall Best: {best['config']}")
        print(f"   Score: {best['score']:.3f} ({best['grade']})")
        
        # Category winners
        fastest = report['champions']['fastest']
        print(f"⚡ Fastest: {fastest['config']}")
        print(f"   Avg Response Time: {fastest['time']:.2f} seconds")
        
        most_accurate = report['champions']['most_accurate']
        print(f"🎯 Most Accurate: {most_accurate['config']}")
        print(f"   Accuracy Score: {most_accurate['accuracy']:.3f}")
        
        best_retrieval = report['champions']['best_retrieval']
        print(f"🔍 Best Retrieval: {best_retrieval['config']}")
        print(f"   Retrieval Score: {best_retrieval['retrieval_score']:.3f}")
        
        print("\\n📈 CONFIGURATION PERFORMANCE:")
        print("-" * 50)
        
        # Sort by composite score
        df_sorted = df.sort_values('composite_score', ascending=False)
        
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            config_name = f"{row['chunking']} + {row['embeddings']} + {row['retrieval']}"
            print(f"{i}. {config_name}")
            print(f"   Overall: {row['composite_score']:.3f} ({row['grade']})")
            print(f"   Quality: {row['quality_score']:.3f} | Accuracy: {row['accuracy_score']:.3f}")
            print(f"   Speed: {row['avg_response_time']:.2f}s | Hallucination Risk: {row['hallucination_risk']:.3f}")
            print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS:")
        print("-" * 50)
        
        top_config = df_sorted.iloc[0]
        if top_config['composite_score'] >= 0.80:
            print("✅ PRODUCTION READY:")
            print(f"   Deploy: {top_config['chunking']} + {top_config['embeddings']} + {top_config['retrieval']}")
            print(f"   Expected Performance: {top_config['grade']} grade")
        elif top_config['composite_score'] >= 0.70:
            print("⚠️ NEEDS OPTIMIZATION:")
            print(f"   Best Available: {top_config['chunking']} + {top_config['embeddings']} + {top_config['retrieval']}")
            print("   Consider additional tuning before production")
        else:
            print("❌ REQUIRES SIGNIFICANT IMPROVEMENT:")
            print("   All configurations scored below acceptable threshold")
            print("   Consider different approaches or additional optimization")


def main():
    """Main entry point."""
    
    print("🔬 RAG Framework - Gemini Pro Comparison")
    print("=" * 50)
    
    # Validate setup
    if not validate_setup():
        print("\\n⛔ Please fix setup issues before running comparison")
        return
    
    # Confirm API key usage
    print(f"\\n💰 Cost Estimate:")
    print(f"   Configurations: {len(TEST_CONFIGURATIONS)}")
    print(f"   Questions per config: 30")
    print(f"   Total Gemini Pro API calls: ~{len(TEST_CONFIGURATIONS) * 30}")
    print(f"   Estimated cost: $0.50 - $2.00 (depending on context length)")
    
    response = input("\\n❓ Proceed with Gemini Pro comparison? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Comparison cancelled")
        return
    
    # Run the comparison
    try:
        run_gemini_comparison()
    except KeyboardInterrupt:
        print("\\n⏹️ Comparison interrupted by user")
    except Exception as e:
        print(f"\\n❌ Error during comparison: {e}")
        print("Check your API key and internet connection")
    
    print(f"\\n📁 Results saved in: {INDUSTRIAL_TEST_CONFIG['output_directory']}")


if __name__ == "__main__":
    main()