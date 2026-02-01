#!/usr/bin/env python3
"""
RAG Pipeline Scoring Summary Display
Shows the final scoring table for all pipeline combinations
"""

import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path
import glob

def display_latest_results():
    """Display the latest test results in a formatted table."""
    
    # Look for the most recent summary file
    output_dir = Path("output")
    if not output_dir.exists():
        print("❌ No output directory found. Run the comprehensive test first.")
        return
    
    # Find the latest summary file
    summary_files = glob.glob(str(output_dir / "comprehensive_rag_test_summary_*.csv"))
    if not summary_files:
        print("❌ No summary files found. Run the comprehensive test first.")
        return
    
    latest_file = max(summary_files, key=os.path.getctime)
    
    try:
        # Load and display the summary
        df = pd.read_csv(latest_file)
        
        print("🎯 RAG PIPELINE PERFORMANCE SUMMARY - ALL COMBINATIONS")
        print("="*80)
        print(f"📅 Results from: {os.path.basename(latest_file)}")
        print(f"📊 Total Combinations Tested: {len(df)}")
        print("="*80)
        
        # Show top 10 performers
        print("🏆 TOP 10 PERFORMING PIPELINES:")
        print("-"*80)
        top_10 = df.head(10)
        
        # Create a more compact display format
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            chunking = row.get('Chunking', 'N/A')
            embeddings = row.get('Embeddings', 'N/A') 
            vectordb = row.get('Vector DB', 'N/A')
            
            print(f"{i:2d}. {row['Avg Score']} | {chunking:12s} + {embeddings:15s} + {vectordb:8s} | {row['Success Rate']:6s}")
        
        print("\\n📈 COMPONENT ANALYSIS:")
        print("-"*40)
        
        # Extract numerical scores for analysis
        if 'Avg Score' in df.columns:
            avg_scores = []
            for score_str in df['Avg Score']:
                try:
                    score = float(score_str.split('/')[0])
                    avg_scores.append(score)
                except:
                    avg_scores.append(0)
        else:
            # Fallback for older format
            avg_scores = []
            for score_str in df['Average Score']:
                try:
                    score = float(score_str.split('/')[0])
                    avg_scores.append(score)
                except:
                    avg_scores.append(0)
        
        df['_numeric_score'] = avg_scores
        
        best_pipeline = df.loc[df['_numeric_score'].idxmax()]
        worst_pipeline = df.loc[df['_numeric_score'].idxmin()]
        
        # Use flexible column names
        pipeline_col = 'Pipeline' if 'Pipeline' in df.columns else df.columns[0]
        score_col = 'Avg Score' if 'Avg Score' in df.columns else 'Average Score'
        success_col = 'Success Rate' if 'Success Rate' in df.columns else df.columns[-2]
        
        print(f"🏆 Best Overall: {best_pipeline[pipeline_col]}")
        print(f"   Score: {best_pipeline[score_col]}")
        print(f"   Success Rate: {best_pipeline[success_col]}")
        
        if 'Chunking' in df.columns:
            print(f"   Components: {best_pipeline['Chunking']} + {best_pipeline['Embeddings']} + {best_pipeline['Vector DB']}")
        
        max_score_col = 'Max Score' if 'Max Score' in df.columns else score_col
        print(f"\\n📈 Highest Single Score: {df[max_score_col].iloc[0]}")
        print(f"📉 Lowest Performer: {worst_pipeline[pipeline_col]} ({worst_pipeline[score_col]})")
        
        # Component analysis if columns are available
        if all(col in df.columns for col in ['Chunking', 'Embeddings', 'Vector DB']):
            print("\\n🧩 BEST COMPONENTS:")
            
            # Analyze chunking strategies
            chunking_performance = df.groupby('Chunking')['_numeric_score'].mean().sort_values(ascending=False)
            print(f"   Best Chunking: {chunking_performance.index[0]} (avg: {chunking_performance.iloc[0]:.2f})")
            
            # Analyze embeddings
            embedding_performance = df.groupby('Embeddings')['_numeric_score'].mean().sort_values(ascending=False)  
            print(f"   Best Embeddings: {embedding_performance.index[0]} (avg: {embedding_performance.iloc[0]:.2f})")
            
            # Analyze vector databases
            vectordb_performance = df.groupby('Vector DB')['_numeric_score'].mean().sort_values(ascending=False)
            print(f"   Best Vector DB: {vectordb_performance.index[0]} (avg: {vectordb_performance.iloc[0]:.2f})")
        
        # Success rate analysis
        success_rates = []
        for rate_str in df['Success Rate']:
            try:
                rate = float(rate_str.rstrip('%'))
                success_rates.append(rate)
            except:
                success_rates.append(0)
        
        avg_success_rate = sum(success_rates) / len(success_rates)
        print(f"📊 Average Success Rate: {avg_success_rate:.1f}%")
        
        print("\\n" + "="*80)
        print("✅ For detailed results, check:")
        print(f"   📄 Summary: {latest_file}")
        print(f"   📄 Detailed: {latest_file.replace('summary', 'detailed')}")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error reading results: {e}")

def show_question_breakdown():
    """Show performance breakdown by question."""
    
    output_dir = Path("output")
    detailed_files = glob.glob(str(output_dir / "comprehensive_rag_test_detailed_*.csv"))
    
    if not detailed_files:
        print("❌ No detailed results found.")
        return
    
    latest_detailed = max(detailed_files, key=os.path.getctime)
    
    try:
        df = pd.read_csv(latest_detailed)
        successful_df = df[df['success'] == True]
        
        if successful_df.empty:
            print("❌ No successful test results found.")
            return
        
        print("\\n📝 QUESTION-BY-QUESTION PERFORMANCE")
        print("="*80)
        
        question_summary = []
        for question_id in successful_df['question_id'].unique():
            q_results = successful_df[successful_df['question_id'] == question_id]
            
            question_summary.append({
                'Question ID': question_id,
                'Average Score': f"{q_results['total_score'].mean():.2f}/10",
                'Best Score': f"{q_results['total_score'].max():.2f}/10",
                'Best Pipeline': q_results.loc[q_results['total_score'].idxmax(), 'pipeline'],
                'Tests Passed': f"{len(q_results)}/{len(df[df['question_id'] == question_id])}"
            })
        
        q_df = pd.DataFrame(question_summary)
        print(q_df.to_string(index=False))
        
        print("\\n🎯 QUESTION DIFFICULTY ANALYSIS")
        print("-"*40)
        
        # Sort by average score to show difficulty
        q_df['_avg_score'] = q_df['Average Score'].str.extract('(\\d+\\.\\d+)').astype(float)
        q_df_sorted = q_df.sort_values('_avg_score', ascending=False)
        
        print(f"📈 Easiest Question: {q_df_sorted.iloc[0]['Question ID']} (Avg: {q_df_sorted.iloc[0]['Average Score']})")
        print(f"📉 Hardest Question: {q_df_sorted.iloc[-1]['Question ID']} (Avg: {q_df_sorted.iloc[-1]['Average Score']})")
        
    except Exception as e:
        print(f"❌ Error analyzing questions: {e}")

def main():
    """Display comprehensive results summary."""
    display_latest_results()
    show_question_breakdown()

if __name__ == "__main__":
    main()