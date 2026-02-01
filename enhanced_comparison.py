"""Enhanced RAG comparison runner with concrete metrics for technical documents."""

import json
import pandas as pd
import numpy as np
import time
import psutil
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
from llm import OpenAILLM, LocalLLM, GeminiProLLM
from evaluation.concrete_metrics import RAGComparisonMetrics


class EnhancedRAGComparator:
    """Enhanced RAG pipeline comparator with comprehensive metrics."""
    
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
        if config.get('llm_type') == 'openai':
            components['llm'] = OpenAILLM(
                api_key=config.get('openai_api_key')
            )
        elif config.get('llm_type') == 'gemini':
            components['llm'] = GeminiProLLM(
                api_key=config.get('gemini_api_key'),
                model=config.get('gemini_model', 'gemini-pro'),
                temperature=config.get('temperature', 0.3),  # Lower temperature for technical content
                max_tokens=config.get('max_tokens', 1000)
            )
        elif config.get('llm_type') == 'local':
            components['llm'] = LocalLLM(
                model_path=config.get('local_model_path')
            )
        else:
            # Default to Gemini Pro if no LLM specified
            components['llm'] = GeminiProLLM(
                api_key=config.get('gemini_api_key', 'your-api-key-here')
            )
        
        return components
    
    def process_document(self, components: Dict[str, Any]) -> None:
        """Process document with given pipeline components."""
        # Chunk the document
        chunks = components['chunker'].chunk(self.document_text)
        print(f\"  📄 Document chunked into {len(chunks)} pieces\")
        
        # Generate embeddings and store in vector DB
        if hasattr(components['retriever'], 'embeddings_model'):
            print(\"  🧮 Generating embeddings...\")
            vectors = components['embeddings'].embed_batch(chunks)
            chunk_ids = [f\"chunk_{i}\" for i in range(len(chunks))]
            
            # Add metadata for better tracking
            metadata = [{\"chunk_index\": i, \"chunk_length\": len(chunk)} for i, chunk in enumerate(chunks)]
            components['vectordb'].add_vectors(vectors, chunk_ids, metadata)
            
        # For BM25, fit on chunks
        if hasattr(components['retriever'], 'fit'):
            print(\"  📊 Fitting BM25 index...\")
            components['retriever'].fit(chunks)
    
    def run_comprehensive_test(self, config: Dict[str, str]) -> Dict[str, Any]:
        """Run comprehensive test with detailed metrics."""
        print(f\"\\n{'='*60}\")
        print(f\"🧪 Testing Configuration: {config}\")
        print(f\"{'='*60}\")
        
        start_time = time.time()
        
        # Setup pipeline
        print(\"🔧 Setting up pipeline components...\")
        components = self.setup_pipeline(config)
        
        # Initialize concrete metrics
        concrete_metrics = RAGComparisonMetrics(components['embeddings'])
        
        # Process document
        print(\"📚 Processing document...\")
        setup_start = time.time()
        self.process_document(components)
        setup_time = time.time() - setup_start
        print(f\"  ✅ Setup completed in {setup_time:.2f} seconds\")
        
        # Test on questions with comprehensive metrics
        print(f\"\\n❓ Running {len(self.test_questions)} test questions...\")
        all_results = []
        process = psutil.Process()
        
        category_stats = {}
        
        for i, question_data in enumerate(self.test_questions, 1):
            question = question_data['question']
            category = question_data['category']
            expected_answer = question_data.get('expected_answer', '')
            
            print(f\"  📝 Question {i}/{len(self.test_questions)}: {category}\")
            
            # Measure memory before
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Measure retrieval
            retrieval_start = time.time()
            try:
                retrieved_docs = components['retriever'].retrieve(question, k=5)
            except Exception as e:
                print(f\"    ⚠️ Retrieval failed: {e}\")
                retrieved_docs = []
                
            retrieval_end = time.time()
            
            # Create context
            if retrieved_docs and isinstance(retrieved_docs[0], dict) and 'text' in retrieved_docs[0]:
                context = \"\\n\".join([doc['text'] for doc in retrieved_docs])\n            elif retrieved_docs:\n                context = \"\\n\".join([str(doc) for doc in retrieved_docs])\n            else:\n                context = \"No relevant context found.\"\n            \n            # Generate response\n            prompt = f\"\"\"Based on the following context, please answer the question accurately.\n            \nContext: {context}\n            \nQuestion: {question}\n            \nAnswer:\"\"\"\n            \n            generation_start = time.time()\n            try:\n                response = components['llm'].generate(prompt)\n            except Exception as e:\n                print(f\"    ⚠️ Generation failed: {e}\")\n                response = \"Unable to generate response.\"\n                \n            generation_end = time.time()\n            \n            # Measure memory after\n            memory_after = process.memory_info().rss / 1024 / 1024  # MB\n            \n            # Calculate comprehensive metrics\n            try:\n                retrieval_metrics = concrete_metrics.calculate_retrieval_metrics(\n                    question, retrieved_docs, [expected_answer] if expected_answer else []\n                )\n                \n                quality_metrics = concrete_metrics.calculate_answer_quality_metrics(\n                    response, expected_answer, context\n                )\n                \n                accuracy_metrics = concrete_metrics.calculate_factual_accuracy_metrics(\n                    response, context\n                )\n                \n                performance_metrics = concrete_metrics.calculate_performance_metrics(\n                    retrieval_start, generation_end, memory_before, memory_after, len(response.split())\n                )\n                \n                composite_scores = concrete_metrics.calculate_composite_score(\n                    retrieval_metrics, quality_metrics, accuracy_metrics, performance_metrics\n                )\n                \n            except Exception as e:\n                print(f\"    ⚠️ Metrics calculation failed: {e}\")\n                # Use default values if metrics calculation fails\n                retrieval_metrics = {'precision_at_5': 0, 'recall_at_5': 0, 'mrr': 0, 'avg_retrieval_score': 0}\n                quality_metrics = {'f1_score': 0, 'bleu_score': 0, 'semantic_similarity': 0, 'answer_completeness': 0, 'context_adherence': 0}\n                accuracy_metrics = {'entity_accuracy': 0, 'numerical_accuracy': 0, 'factual_consistency': 0, 'hallucination_risk': 1}\n                performance_metrics = {'response_time': generation_end - retrieval_start, 'memory_usage_mb': 0, 'tokens_per_second': 0, 'efficiency_score': 0}\n                composite_scores = {'retrieval_score': 0, 'quality_score': 0, 'accuracy_score': 0, 'performance_score': 0, 'composite_score': 0, 'grade': 'F'}\n            \n            # Store detailed results\n            question_result = {\n                'question_id': question_data['id'],\n                'category': category,\n                'question': question,\n                'predicted_answer': response,\n                'expected_answer': expected_answer,\n                'context_preview': context[:300] + '...' if len(context) > 300 else context,\n                'retrieval_metrics': retrieval_metrics,\n                'quality_metrics': quality_metrics,\n                'accuracy_metrics': accuracy_metrics,\n                'performance_metrics': performance_metrics,\n                'composite_scores': composite_scores\n            }\n            \n            all_results.append(question_result)\n            \n            # Track category performance\n            if category not in category_stats:\n                category_stats[category] = []\n            category_stats[category].append(composite_scores['composite_score'])\n            \n            # Show progress\n            print(f\"    📊 Score: {composite_scores['composite_score']:.3f} ({composite_scores['grade']})\")\n        \n        # Calculate aggregate metrics\n        total_time = time.time() - start_time\n        \n        # Aggregate all metrics\n        avg_retrieval_score = np.mean([r['composite_scores']['retrieval_score'] for r in all_results])\n        avg_quality_score = np.mean([r['composite_scores']['quality_score'] for r in all_results])\n        avg_accuracy_score = np.mean([r['composite_scores']['accuracy_score'] for r in all_results])\n        avg_performance_score = np.mean([r['composite_scores']['performance_score'] for r in all_results])\n        avg_composite_score = np.mean([r['composite_scores']['composite_score'] for r in all_results])\n        \n        avg_response_time = np.mean([r['performance_metrics']['response_time'] for r in all_results])\n        avg_memory_usage = np.mean([r['performance_metrics']['memory_usage_mb'] for r in all_results])\n        avg_hallucination_risk = np.mean([r['accuracy_metrics']['hallucination_risk'] for r in all_results])\n        \n        # Calculate category performance\n        category_performance = {}\n        for category, scores in category_stats.items():\n            category_performance[category] = {\n                'avg_score': np.mean(scores),\n                'min_score': np.min(scores),\n                'max_score': np.max(scores),\n                'question_count': len(scores)\n            }\n        \n        # Determine overall grade\n        overall_grade = concrete_metrics._score_to_grade(avg_composite_score)\n        \n        print(f\"\\n📊 Results Summary:\")\n        print(f\"  Overall Score: {avg_composite_score:.3f} ({overall_grade})\")\n        print(f\"  Avg Response Time: {avg_response_time:.2f}s\")\n        print(f\"  Hallucination Risk: {avg_hallucination_risk:.3f}\")\n        \n        return {\n            'config': config,\n            'detailed_results': all_results,\n            'category_performance': category_performance,\n            'aggregate_metrics': {\n                'avg_retrieval_score': avg_retrieval_score,\n                'avg_quality_score': avg_quality_score,\n                'avg_accuracy_score': avg_accuracy_score,\n                'avg_performance_score': avg_performance_score,\n                'avg_composite_score': avg_composite_score,\n                'overall_grade': overall_grade\n            },\n            'performance_summary': {\n                'avg_response_time': avg_response_time,\n                'avg_memory_usage_mb': avg_memory_usage,\n                'avg_hallucination_risk': avg_hallucination_risk,\n                'total_processing_time': total_time,\n                'setup_time': setup_time\n            },\n            'timestamp': datetime.now().isoformat()\n        }\n    \n    def compare_configurations(self, configurations: List[Dict[str, str]]) -> pd.DataFrame:\n        \"\"\"Compare multiple configurations with enhanced metrics.\"\"\"\n        results = []\n        \n        print(f\"🚀 Starting Enhanced RAG Pipeline Comparison\")\n        print(f\"📋 Testing {len(configurations)} configurations on {len(self.test_questions)} questions\")\n        print(f\"📄 Document: {self.document_path}\")\n        \n        for i, config in enumerate(configurations, 1):\n            print(f\"\\n{'🔬' * 20} Test {i}/{len(configurations)} {'🔬' * 20}\")\n            \n            try:\n                result = self.run_comprehensive_test(config)\n                results.append(result)\n                \n            except Exception as e:\n                print(f\"❌ Configuration failed: {config}\")\n                print(f\"   Error: {str(e)}\")\n                continue\n        \n        # Convert to DataFrame for comparison\n        comparison_data = []\n        for result in results:\n            config = result['config']\n            metrics = result['aggregate_metrics']\n            perf = result['performance_summary']\n            \n            row = {\n                'chunking': config['chunking'],\n                'embeddings': config['embeddings'], \n                'retrieval': config['retrieval'],\n                'composite_score': metrics['avg_composite_score'],\n                'grade': metrics['overall_grade'],\n                'retrieval_score': metrics['avg_retrieval_score'],\n                'quality_score': metrics['avg_quality_score'],\n                'accuracy_score': metrics['avg_accuracy_score'],\n                'performance_score': metrics['avg_performance_score'],\n                'avg_response_time': perf['avg_response_time'],\n                'hallucination_risk': perf['avg_hallucination_risk'],\n                'total_time': perf['total_processing_time']\n            }\n            comparison_data.append(row)\n        \n        df = pd.DataFrame(comparison_data)\n        return df, results\n    \n    def generate_detailed_report(self, df: pd.DataFrame, results: List[Dict], output_path: str):\n        \"\"\"Generate comprehensive comparison report.\"\"\"\n        \n        # Find best configurations\n        if len(df) > 0:\n            best_overall = df.loc[df['composite_score'].idxmax()]\n            fastest = df.loc[df['avg_response_time'].idxmin()]\n            most_accurate = df.loc[df['accuracy_score'].idxmax()]\n            best_retrieval = df.loc[df['retrieval_score'].idxmax()]\n        else:\n            print(\"❌ No successful configurations to analyze\")\n            return\n        \n        report = {\n            'summary': {\n                'total_configurations_tested': len(df),\n                'total_questions': len(self.test_questions),\n                'document_path': self.document_path,\n                'test_categories': list(set([q['category'] for q in self.test_questions])),\n                'generated_at': datetime.now().isoformat()\n            },\n            'champions': {\n                'overall_best': {\n                    'config': f\"{best_overall['chunking']} + {best_overall['embeddings']} + {best_overall['retrieval']}\",\n                    'score': float(best_overall['composite_score']),\n                    'grade': best_overall['grade']\n                },\n                'fastest': {\n                    'config': f\"{fastest['chunking']} + {fastest['embeddings']} + {fastest['retrieval']}\",\n                    'time': float(fastest['avg_response_time'])\n                },\n                'most_accurate': {\n                    'config': f\"{most_accurate['chunking']} + {most_accurate['embeddings']} + {most_accurate['retrieval']}\",\n                    'accuracy': float(most_accurate['accuracy_score'])\n                },\n                'best_retrieval': {\n                    'config': f\"{best_retrieval['chunking']} + {best_retrieval['embeddings']} + {best_retrieval['retrieval']}\",\n                    'retrieval_score': float(best_retrieval['retrieval_score'])\n                }\n            },\n            'detailed_results': results,\n            'comparison_table': df.to_dict('records')\n        }\n        \n        # Save comprehensive report\n        os.makedirs(os.path.dirname(output_path), exist_ok=True)\n        with open(output_path, 'w', encoding='utf-8') as f:\n            json.dump(report, f, indent=2, ensure_ascii=False)\n        \n        # Save CSV for analysis\n        csv_path = output_path.replace('.json', '.csv')\n        df.to_csv(csv_path, index=False)\n        \n        print(f\"\\n📊 Detailed report saved to: {output_path}\")\n        print(f\"📋 CSV data saved to: {csv_path}\")\n        \n        return report\n\n\ndef main():\n    \"\"\"Main enhanced comparison runner.\"\"\"\n    # Define test configurations for industrial document\n    test_configurations = [\n        # Chunking strategy comparison (good for table extraction)\n        {\"chunking\": \"fixed\", \"embeddings\": \"sentence_transformers\", \"retrieval\": \"dense\"},\n        {\"chunking\": \"semantic\", \"embeddings\": \"sentence_transformers\", \"retrieval\": \"dense\"},\n        {\"chunking\": \"structure_aware\", \"embeddings\": \"sentence_transformers\", \"retrieval\": \"dense\"},\n        \n        # Embedding model comparison (technical content understanding)\n        {\"chunking\": \"structure_aware\", \"embeddings\": \"sentence_transformers\", \"retrieval\": \"dense\"},\n        {\"chunking\": \"structure_aware\", \"embeddings\": \"bge\", \"retrieval\": \"dense\"},\n        \n        # Retrieval strategy comparison (table vs text content)\n        {\"chunking\": \"structure_aware\", \"embeddings\": \"bge\", \"retrieval\": \"dense\"},\n        {\"chunking\": \"structure_aware\", \"embeddings\": \"bge\", \"retrieval\": \"hybrid\"},\n    ]\n    \n    # Initialize enhanced comparator\n    document_path = \"data/marker_md/RedBook_Markdown.md\"\n    questions_path = \"data/test_questions.json\"\n    \n    if not os.path.exists(document_path):\n        print(f\"❌ Document not found: {document_path}\")\n        print(\"Please copy your RedBook_Markdown.md file to the data/marker_md/ folder\")\n        return\n    \n    comparator = EnhancedRAGComparator(document_path, questions_path)\n    \n    # Run enhanced comparison\n    print(\"🚀 Starting Enhanced RAG Pipeline Comparison for Industrial Document...\")\n    df, results = comparator.compare_configurations(test_configurations)\n    \n    if len(df) == 0:\n        print(\"❌ No configurations completed successfully\")\n        return\n    \n    # Generate comprehensive report\n    timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n    output_path = f\"results/enhanced_comparison_report_{timestamp}.json\"\n    report = comparator.generate_detailed_report(df, results, output_path)\n    \n    # Print final summary\n    print(\"\\n\" + \"=\"*80)\n    print(\"🏆 ENHANCED COMPARISON SUMMARY\")\n    print(\"=\"*80)\n    print(f\"📋 Configurations Tested: {len(df)}\")\n    print(f\"❓ Questions Tested: {len(comparator.test_questions)}\")\n    \n    if len(df) > 0:\n        best = report['champions']['overall_best']\n        print(f\"🥇 Overall Winner: {best['config']}\")\n        print(f\"   Score: {best['score']:.3f} ({best['grade']})\")\n        \n        fastest = report['champions']['fastest']\n        print(f\"⚡ Fastest: {fastest['config']} ({fastest['time']:.2f}s avg)\")\n        \n        accurate = report['champions']['most_accurate']\n        print(f\"🎯 Most Accurate: {accurate['config']} ({accurate['accuracy']:.3f})\")\n    \n    print(f\"\\n📊 Full results available in: results/\")\n\n\nif __name__ == \"__main__\":\n    main()