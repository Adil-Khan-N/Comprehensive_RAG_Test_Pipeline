"""Automated comparison runner for different RAG configurations."""

import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Any
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chunking import FixedSizeChunker, SemanticChunker, StructureAwareChunker, HybridChunker
from embeddings import OpenAIEmbeddings, SentenceTransformersEmbeddings, BGEEmbeddings
from vectordb import FAISSVectorDB, ChromaVectorDB
from retriever import DenseRetriever, BM25Retriever, HybridRetriever
from llm import OpenAILLM, LocalLLM
from evaluation import RAGMetrics, HallucinationDetector
from evaluation.concrete_metrics import RAGComparisonMetrics
from evaluation.concrete_metrics import RAGComparisonMetrics


class RAGComparator:
    """Compare different RAG pipeline configurations."""
    
    def __init__(self, document_path: str, questions_path: str):
        self.document_path = document_path
        self.questions_path = questions_path
        self.test_questions = self.load_questions()
        self.document_text = self.load_document()
        self.results = []
        
    def load_questions(self) -> List[Dict]:
        """Load test questions from JSON file."""
        with open(self.questions_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def load_document(self) -> str:
        """Load the markdown document."""
        with open(self.document_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def setup_pipeline(self, config: Dict[str, str]) -> Dict[str, Any]:
        """Setup a RAG pipeline based on configuration."""
        components = {}
        
        # Setup chunker
        if config['chunking'] == 'fixed':
            components['chunker'] = FixedSizeChunker(chunk_size=512, overlap=50)
        elif config['chunking'] == 'semantic':
            components['chunker'] = SemanticChunker()
        elif config['chunking'] == 'structure_aware':
            components['chunker'] = StructureAwareChunker()
        elif config['chunking'] == 'hybrid':
            components['chunker'] = HybridChunker()
            
        # Setup embeddings
        if config['embeddings'] == 'openai':
            components['embeddings'] = OpenAIEmbeddings()
        elif config['embeddings'] == 'sentence_transformers':
            components['embeddings'] = SentenceTransformersEmbeddings()
        elif config['embeddings'] == 'bge':
            components['embeddings'] = BGEEmbeddings()
            
        # Setup vector database
        components['vectordb'] = FAISSVectorDB(dimension=768)  # Default dimension
        
        # Setup retriever
        if config['retrieval'] == 'dense':
            components['retriever'] = DenseRetriever(components['embeddings'], components['vectordb'])
        elif config['retrieval'] == 'bm25':
            components['retriever'] = BM25Retriever()
        elif config['retrieval'] == 'hybrid':
            dense_ret = DenseRetriever(components['embeddings'], components['vectordb'])
            bm25_ret = BM25Retriever()
            components['retriever'] = HybridRetriever(dense_ret, bm25_ret)
            
        # Setup LLM
        components['llm'] = OpenAILLM()
        
        return components
    
    def process_document(self, components: Dict[str, Any]) -> None:
        """Process document with given pipeline components."""
        # Chunk the document
        chunks = components['chunker'].chunk(self.document_text)
        
        # Generate embeddings and store in vector DB
        if hasattr(components['retriever'], 'embeddings_model'):
            vectors = components['embeddings'].embed_batch(chunks)
            chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
            components['vectordb'].add_vectors(vectors, chunk_ids)
            
        # For BM25, fit on chunks
        if hasattr(components['retriever'], 'fit'):
            components['retriever'].fit(chunks)
    
    def run_single_test(self, config: Dict[str, str]) -> Dict[str, Any]:
        """Run a single configuration test."""
        print(f"Testing configuration: {config}")
        start_time = time.time()
        
        # Setup pipeline
        components = self.setup_pipeline(config)
        
        # Process document
        self.process_document(components)
        
        # Test on questions
        predictions = []
        ground_truths = []
        response_times = []
        retrieval_scores = []
        
        for question_data in self.test_questions:
            question = question_data['question']
            
            # Measure retrieval time
            retrieval_start = time.time()
            retrieved_docs = components['retriever'].retrieve(question, k=5)
            retrieval_time = time.time() - retrieval_start
            
            # Create context
            if isinstance(retrieved_docs[0], dict) and 'text' in retrieved_docs[0]:
                context = "\\n".join([doc['text'] for doc in retrieved_docs])
            else:
                context = "\\n".join([str(doc) for doc in retrieved_docs])
            
            # Generate response
            prompt = f"Context: {context}\\n\\nQuestion: {question}\\n\\nAnswer:"
            generation_start = time.time()
            response = components['llm'].generate(prompt)
            generation_time = time.time() - generation_start
            
            predictions.append(response)
            ground_truths.append(question_data.get('expected_answer', ''))  # Placeholder
            response_times.append(retrieval_time + generation_time)
            
            # Calculate average retrieval score
            if retrieved_docs and isinstance(retrieved_docs[0], dict) and 'score' in retrieved_docs[0]:
                avg_score = np.mean([doc['score'] for doc in retrieved_docs])
                retrieval_scores.append(avg_score)
            else:
                retrieval_scores.append(0.0)
        
        # Calculate metrics
        metrics = RAGMetrics()
        # Note: We don't have ground truth answers, so we'll focus on other metrics
        
        # Hallucination detection
        hallucination_detector = HallucinationDetector()
        hallucination_results = []
        
        for pred, question_data in zip(predictions, self.test_questions):
            # Use retrieved context for hallucination detection
            retrieved_docs = components['retriever'].retrieve(question_data['question'], k=5)
            context = "\\n".join([str(doc) for doc in retrieved_docs])
            halluc_result = hallucination_detector.comprehensive_hallucination_check(pred, context)
            hallucination_results.append(halluc_result['overall_hallucination_score'])
        
        total_time = time.time() - start_time
        
        return {
            'config': config,
            'predictions': predictions,
            'avg_response_time': np.mean(response_times),
            'avg_retrieval_score': np.mean(retrieval_scores) if retrieval_scores else 0.0,
            'avg_hallucination_score': np.mean(hallucination_results),
            'total_processing_time': total_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def compare_configurations(self, configurations: List[Dict[str, str]]) -> pd.DataFrame:
        """Compare multiple configurations and return results."""
        results = []
        
        for i, config in enumerate(configurations, 1):
            print(f"\\n{'='*50}")
            print(f"Running Test {i}/{len(configurations)}")
            print(f"{'='*50}")
            
            try:
                result = self.run_single_test(config)
                results.append(result)
                
                # Print immediate results
                print(f"✅ Completed: {config}")
                print(f"   Avg Response Time: {result['avg_response_time']:.2f}s")
                print(f"   Avg Retrieval Score: {result['avg_retrieval_score']:.3f}")
                print(f"   Avg Hallucination Score: {result['avg_hallucination_score']:.3f}")
                
            except Exception as e:
                print(f"❌ Failed: {config}")
                print(f"   Error: {str(e)}")
                continue
        
        # Convert to DataFrame for easy comparison
        comparison_data = []
        for result in results:
            row = {
                'chunking': result['config']['chunking'],
                'embeddings': result['config']['embeddings'], 
                'retrieval': result['config']['retrieval'],
                'avg_response_time': result['avg_response_time'],
                'avg_retrieval_score': result['avg_retrieval_score'],
                'avg_hallucination_score': result['avg_hallucination_score'],
                'total_processing_time': result['total_processing_time']
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df, results
    
    def generate_comparison_report(self, df: pd.DataFrame, results: List[Dict], output_path: str):
        """Generate a comprehensive comparison report."""
        report = {
            'summary': {
                'total_configurations_tested': len(df),
                'test_questions_count': len(self.test_questions),
                'document_path': self.document_path,
                'generated_at': datetime.now().isoformat()
            },
            'rankings': {
                'fastest_response': df.loc[df['avg_response_time'].idxmin()].to_dict(),
                'best_retrieval': df.loc[df['avg_retrieval_score'].idxmax()].to_dict(),
                'least_hallucination': df.loc[df['avg_hallucination_score'].idxmin()].to_dict()
            },
            'detailed_results': results,
            'comparison_table': df.to_dict('records')
        }
        
        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save CSV for easy analysis
        csv_path = output_path.replace('.json', '.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"\\n📊 Comparison report saved to: {output_path}")
        print(f"📋 CSV data saved to: {csv_path}")
        
        return report


def main():
    """Main comparison runner."""
    # Define test configurations
    test_configurations = [
        # Chunking strategy comparison
        {"chunking": "fixed", "embeddings": "sentence_transformers", "retrieval": "dense"},
        {"chunking": "semantic", "embeddings": "sentence_transformers", "retrieval": "dense"},
        {"chunking": "structure_aware", "embeddings": "sentence_transformers", "retrieval": "dense"},
        {"chunking": "hybrid", "embeddings": "sentence_transformers", "retrieval": "dense"},
        
        # Embedding model comparison (using best chunking from above)
        {"chunking": "semantic", "embeddings": "sentence_transformers", "retrieval": "dense"},
        {"chunking": "semantic", "embeddings": "bge", "retrieval": "dense"},
        
        # Retrieval strategy comparison
        {"chunking": "semantic", "embeddings": "bge", "retrieval": "dense"},
        {"chunking": "semantic", "embeddings": "bge", "retrieval": "hybrid"},
    ]
    
    # Initialize comparator
    document_path = "data/marker_md/RedBook_Markdown.md"
    questions_path = "data/test_questions.json"
    
    comparator = RAGComparator(document_path, questions_path)
    
    # Run comparison
    print("🚀 Starting RAG Pipeline Comparison...")
    df, results = comparator.compare_configurations(test_configurations)
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/comparison_report_{timestamp}.json"
    report = comparator.generate_comparison_report(df, results, output_path)
    
    # Print summary
    print("\\n" + "="*60)
    print("🏆 COMPARISON SUMMARY")
    print("="*60)
    print(f"Total Configurations Tested: {len(df)}")
    print("\\nTop Performers:")
    print(f"⚡ Fastest Response: {report['rankings']['fastest_response']['chunking']} + {report['rankings']['fastest_response']['embeddings']} + {report['rankings']['fastest_response']['retrieval']}")
    print(f"🎯 Best Retrieval: {report['rankings']['best_retrieval']['chunking']} + {report['rankings']['best_retrieval']['embeddings']} + {report['rankings']['best_retrieval']['retrieval']}")
    print(f"🛡️ Least Hallucination: {report['rankings']['least_hallucination']['chunking']} + {report['rankings']['least_hallucination']['embeddings']} + {report['rankings']['least_hallucination']['retrieval']}")
    
    print("\\n📊 Detailed results saved to JSON and CSV files.")


if __name__ == "__main__":
    main()