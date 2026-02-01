"""Generate visual comparison dashboard for RAG pipeline results."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from datetime import datetime
import os


class ComparisonVisualizer:
    """Create visualizations for RAG pipeline comparisons."""
    
    def __init__(self, results_path: str):
        self.results_path = results_path
        self.report = self.load_results()
        self.df = pd.DataFrame(self.report['comparison_table'])
        
    def load_results(self) -> dict:
        """Load comparison results from JSON file."""
        with open(self.results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_performance_comparison(self, save_path: str = None):
        """Create performance comparison charts."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RAG Pipeline Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Response Time Comparison
        ax1 = axes[0, 0]
        self.df.plot(x='chunking', y='avg_response_time', kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Average Response Time by Chunking Strategy')
        ax1.set_ylabel('Response Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Retrieval Score Comparison
        ax2 = axes[0, 1]
        self.df.plot(x='embeddings', y='avg_retrieval_score', kind='bar', ax=ax2, color='lightgreen')
        ax2.set_title('Average Retrieval Score by Embedding Model')
        ax2.set_ylabel('Retrieval Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Hallucination Score Comparison (lower is better)
        ax3 = axes[1, 0]
        self.df.plot(x='retrieval', y='avg_hallucination_score', kind='bar', ax=ax3, color='lightcoral')
        ax3.set_title('Average Hallucination Score by Retrieval Method')
        ax3.set_ylabel('Hallucination Score (lower is better)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Processing Time Comparison
        ax4 = axes[1, 1]
        configs = [f"{row['chunking'][:4]}+{row['embeddings'][:4]}+{row['retrieval'][:4]}" 
                  for _, row in self.df.iterrows()]
        ax4.bar(range(len(configs)), self.df['total_processing_time'], color='gold')
        ax4.set_title('Total Processing Time by Configuration')
        ax4.set_ylabel('Processing Time (seconds)')
        ax4.set_xticks(range(len(configs)))
        ax4.set_xticklabels(configs, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance comparison saved to: {save_path}")
        
        plt.show()
    
    def create_heatmap_comparison(self, save_path: str = None):
        """Create a heatmap showing all metrics across configurations."""
        # Prepare data for heatmap
        metrics = ['avg_response_time', 'avg_retrieval_score', 'avg_hallucination_score', 'total_processing_time']
        config_names = [f"{row['chunking'][:4]}+{row['embeddings'][:4]}+{row['retrieval'][:4]}" 
                       for _, row in self.df.iterrows()]
        
        # Normalize metrics for better visualization (0-1 scale)
        heatmap_data = self.df[metrics].copy()
        for col in metrics:
            if col == 'avg_hallucination_score':  # Lower is better
                heatmap_data[col] = 1 - (heatmap_data[col] - heatmap_data[col].min()) / (heatmap_data[col].max() - heatmap_data[col].min())
            elif col in ['avg_response_time', 'total_processing_time']:  # Lower is better
                heatmap_data[col] = 1 - (heatmap_data[col] - heatmap_data[col].min()) / (heatmap_data[col].max() - heatmap_data[col].min())
            else:  # Higher is better
                heatmap_data[col] = (heatmap_data[col] - heatmap_data[col].min()) / (heatmap_data[col].max() - heatmap_data[col].min())
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data.T, 
                   xticklabels=config_names,
                   yticklabels=['Response Time\\n(lower better)', 'Retrieval Score\\n(higher better)', 
                               'Hallucination\\n(lower better)', 'Processing Time\\n(lower better)'],
                   annot=True, 
                   cmap='RdYlGn',
                   center=0.5,
                   fmt='.2f')
        
        plt.title('RAG Pipeline Performance Heatmap\\n(Green = Better Performance)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Configuration (Chunking + Embeddings + Retrieval)')
        plt.ylabel('Performance Metrics (Normalized 0-1)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap comparison saved to: {save_path}")
        
        plt.show()
    
    def create_radar_chart(self, save_path: str = None):
        """Create radar charts for top 3 configurations."""
        # Get top 3 configurations based on average normalized performance
        metrics = ['avg_response_time', 'avg_retrieval_score', 'avg_hallucination_score', 'total_processing_time']
        
        # Calculate overall performance score
        perf_scores = []
        for _, row in self.df.iterrows():
            # Normalize each metric (higher score = better performance)
            norm_response_time = 1 - (row['avg_response_time'] - self.df['avg_response_time'].min()) / (self.df['avg_response_time'].max() - self.df['avg_response_time'].min())
            norm_retrieval = (row['avg_retrieval_score'] - self.df['avg_retrieval_score'].min()) / (self.df['avg_retrieval_score'].max() - self.df['avg_retrieval_score'].min())
            norm_halluc = 1 - (row['avg_hallucination_score'] - self.df['avg_hallucination_score'].min()) / (self.df['avg_hallucination_score'].max() - self.df['avg_hallucination_score'].min())
            norm_process_time = 1 - (row['total_processing_time'] - self.df['total_processing_time'].min()) / (self.df['total_processing_time'].max() - self.df['total_processing_time'].min())
            
            overall_score = (norm_response_time + norm_retrieval + norm_halluc + norm_process_time) / 4
            perf_scores.append(overall_score)
        
        self.df['overall_performance'] = perf_scores
        top_3 = self.df.nlargest(3, 'overall_performance')
        
        # Create radar chart
        categories = ['Response\\nTime', 'Retrieval\\nQuality', 'Factual\\nAccuracy', 'Processing\\nSpeed']
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, (_, config) in enumerate(top_3.iterrows()):
            values = [
                1 - (config['avg_response_time'] - self.df['avg_response_time'].min()) / (self.df['avg_response_time'].max() - self.df['avg_response_time'].min()),
                (config['avg_retrieval_score'] - self.df['avg_retrieval_score'].min()) / (self.df['avg_retrieval_score'].max() - self.df['avg_retrieval_score'].min()),
                1 - (config['avg_hallucination_score'] - self.df['avg_hallucination_score'].min()) / (self.df['avg_hallucination_score'].max() - self.df['avg_hallucination_score'].min()),
                1 - (config['total_processing_time'] - self.df['total_processing_time'].min()) / (self.df['total_processing_time'].max() - self.df['total_processing_time'].min())
            ]
            values += values[:1]  # Complete the circle
            
            label = f"{config['chunking']} + {config['embeddings']} + {config['retrieval']}"
            ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Top 3 RAG Configurations - Performance Radar\\n(Outer edge = Better Performance)', 
                    size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Radar chart saved to: {save_path}")
        
        plt.show()
    
    def generate_summary_report(self, output_path: str):
        """Generate a summary report with recommendations."""
        # Calculate rankings
        best_overall = self.df.loc[self.df['overall_performance'].idxmax()]
        fastest = self.df.loc[self.df['avg_response_time'].idxmin()]
        most_accurate = self.df.loc[self.df['avg_hallucination_score'].idxmin()]
        best_retrieval = self.df.loc[self.df['avg_retrieval_score'].idxmax()]
        
        summary = f"""
# RAG Pipeline Comparison Summary Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🏆 Overall Winner
**Configuration**: {best_overall['chunking']} + {best_overall['embeddings']} + {best_overall['retrieval']}
- Overall Performance Score: {best_overall['overall_performance']:.3f}
- Response Time: {best_overall['avg_response_time']:.2f}s
- Retrieval Score: {best_overall['avg_retrieval_score']:.3f}
- Hallucination Score: {best_overall['avg_hallucination_score']:.3f}

## 📊 Category Leaders

### ⚡ Fastest Response
**{fastest['chunking']} + {fastest['embeddings']} + {fastest['retrieval']}**
- Average Response Time: {fastest['avg_response_time']:.2f} seconds

### 🎯 Best Retrieval Quality  
**{best_retrieval['chunking']} + {best_retrieval['embeddings']} + {best_retrieval['retrieval']}**
- Average Retrieval Score: {best_retrieval['avg_retrieval_score']:.3f}

### 🛡️ Most Factually Accurate
**{most_accurate['chunking']} + {most_accurate['embeddings']} + {most_accurate['retrieval']}**
- Hallucination Score: {most_accurate['avg_hallucination_score']:.3f} (lower is better)

## 💡 Recommendations

### For Production Use:
- **Best Overall**: {best_overall['chunking']} + {best_overall['embeddings']} + {best_overall['retrieval']}
- Provides balanced performance across all metrics

### For Real-time Applications:
- **Fastest**: {fastest['chunking']} + {fastest['embeddings']} + {fastest['retrieval']}
- Prioritizes response speed

### For High-accuracy Requirements:
- **Most Accurate**: {most_accurate['chunking']} + {most_accurate['embeddings']} + {most_accurate['retrieval']}
- Minimizes hallucinations and factual errors

## 📈 Performance Insights

{self.generate_insights()}

## 🔧 Next Steps
1. Run extended testing on the top 3 configurations
2. Test with larger question sets
3. Evaluate on domain-specific metrics
4. Consider ensemble approaches combining best elements
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"Summary report saved to: {output_path}")
    
    def generate_insights(self) -> str:
        """Generate insights from the comparison data."""
        insights = []
        
        # Chunking strategy insights
        chunking_perf = self.df.groupby('chunking')['overall_performance'].mean().sort_values(ascending=False)
        insights.append(f"- **Best Chunking Strategy**: {chunking_perf.index[0]} (avg performance: {chunking_perf.iloc[0]:.3f})")
        
        # Embedding insights
        embedding_perf = self.df.groupby('embeddings')['overall_performance'].mean().sort_values(ascending=False)
        insights.append(f"- **Best Embedding Model**: {embedding_perf.index[0]} (avg performance: {embedding_perf.iloc[0]:.3f})")
        
        # Retrieval insights
        retrieval_perf = self.df.groupby('retrieval')['overall_performance'].mean().sort_values(ascending=False)
        insights.append(f"- **Best Retrieval Method**: {retrieval_perf.index[0]} (avg performance: {retrieval_perf.iloc[0]:.3f})")
        
        return "\\n".join(insights)


def main():
    """Generate all visualizations and reports."""
    # Example usage - replace with your actual results file
    results_file = "results/comparison_report_20260201_120000.json"  # Update with actual filename
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print("Please run compare_implementations.py first to generate comparison results.")
        return
    
    visualizer = ComparisonVisualizer(results_file)
    
    # Create output directory
    os.makedirs("results/visualizations", exist_ok=True)
    
    # Generate all visualizations
    print("📊 Generating performance comparison charts...")
    visualizer.create_performance_comparison("results/visualizations/performance_comparison.png")
    
    print("🔥 Creating performance heatmap...")
    visualizer.create_heatmap_comparison("results/visualizations/performance_heatmap.png")
    
    print("🎯 Generating radar chart for top configurations...")
    visualizer.create_radar_chart("results/visualizations/top_configs_radar.png")
    
    print("📝 Creating summary report...")
    visualizer.generate_summary_report("results/comparison_summary.md")
    
    print("\\n✅ All visualizations and reports generated successfully!")
    print("📁 Check the 'results' folder for all outputs.")


if __name__ == "__main__":
    main()