#!/usr/bin/env python3
"""
Comprehensive RAG Pipeline Testing and Scoring System

This script tests all possible combinations of chunking, embeddings, and vector stores
against technical questions and provides detailed scoring for each pipeline.
"""

import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
import time
import re
from typing import Dict, List, Tuple
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import validate_config, GEMINI_API_KEY, TEST_DOCUMENT_PATH, OUTPUT_DIR
from llm.gemini import GeminiProLLM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'comprehensive_rag_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AdvancedRAGScorer:
    """Advanced scoring system for RAG pipeline responses."""
    
    def __init__(self):
        self.scoring_weights = {
            'keyword_match': 0.4,      # 40% for keyword presence
            'numerical_accuracy': 0.3,  # 30% for numerical values
            'semantic_relevance': 0.2,  # 20% for overall relevance
            'completeness': 0.1         # 10% for answer completeness
        }
    
    def score_response(self, question: Dict, response: str) -> Dict:
        """
        Score a response against expected answer with detailed breakdown.
        
        Returns:
            Dict with detailed scoring breakdown
        """
        max_score = question.get('max_score', 10)
        scoring_keywords = question.get('scoring_keywords', [])
        expected_answer = question.get('expected_answer', '')
        
        scores = {
            'keyword_score': 0,
            'numerical_score': 0,
            'semantic_score': 0,
            'completeness_score': 0,
            'total_score': 0,
            'breakdown': {}
        }
        
        # 1. Keyword matching score
        keyword_matches = 0
        found_keywords = []
        for keyword in scoring_keywords:
            if keyword.lower() in response.lower():
                keyword_matches += 1
                found_keywords.append(keyword)
        
        keyword_score = (keyword_matches / len(scoring_keywords)) * max_score if scoring_keywords else 0
        scores['keyword_score'] = keyword_score * self.scoring_weights['keyword_match']
        scores['breakdown']['found_keywords'] = found_keywords
        
        # 2. Numerical accuracy score
        numerical_score = self._score_numerical_accuracy(expected_answer, response, max_score)
        scores['numerical_score'] = numerical_score * self.scoring_weights['numerical_accuracy']
        
        # 3. Semantic relevance score (simplified)
        semantic_score = self._score_semantic_relevance(question['question'], response, max_score)
        scores['semantic_score'] = semantic_score * self.scoring_weights['semantic_relevance']
        
        # 4. Completeness score
        completeness_score = self._score_completeness(expected_answer, response, max_score)
        scores['completeness_score'] = completeness_score * self.scoring_weights['completeness']
        
        # Calculate total score
        scores['total_score'] = (
            scores['keyword_score'] + 
            scores['numerical_score'] + 
            scores['semantic_score'] + 
            scores['completeness_score']
        )
        
        return scores
    
    def _score_numerical_accuracy(self, expected: str, response: str, max_score: int) -> float:
        """Score numerical accuracy by finding numbers in both texts."""
        expected_numbers = re.findall(r'\d+\.?\d*', expected)
        response_numbers = re.findall(r'\d+\.?\d*', response)
        
        if not expected_numbers:
            return max_score  # No numbers to compare
        
        matches = 0
        for exp_num in expected_numbers:
            for resp_num in response_numbers:
                if abs(float(exp_num) - float(resp_num)) < 0.1:  # Allow small floating point differences
                    matches += 1
                    break
        
        return (matches / len(expected_numbers)) * max_score if expected_numbers else 0
    
    def _score_semantic_relevance(self, question: str, response: str, max_score: int) -> float:
        """Simple semantic relevance scoring based on question keywords in response."""
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common words
        stop_words = {'what', 'is', 'the', 'for', 'in', 'and', 'or', 'of', 'a', 'an', 'to', 'with'}
        question_words -= stop_words
        
        if not question_words:
            return max_score
        
        matches = len(question_words.intersection(response_words))
        return (matches / len(question_words)) * max_score
    
    def _score_completeness(self, expected: str, response: str, max_score: int) -> float:
        """Score based on response length and detail compared to expected answer."""
        if not expected or not response:
            return 0
        
        expected_length = len(expected.split())
        response_length = len(response.split())
        
        # Ideal response should be 50-150% of expected length
        if expected_length == 0:
            return max_score
        
        length_ratio = response_length / expected_length
        if 0.5 <= length_ratio <= 1.5:
            return max_score
        elif length_ratio < 0.5:
            return max_score * length_ratio * 2  # Penalty for too short
        else:
            return max_score * (1.5 / length_ratio)  # Penalty for too long

class ComprehensiveRAGTester:
    """Comprehensive RAG pipeline testing system."""
    
    def __init__(self):
        self.scorer = AdvancedRAGScorer()
        self.test_results = []
        
        # Initialize LLM
        self.llm = GeminiProLLM(api_key=GEMINI_API_KEY)
        
        # Define all available components
        self.chunking_strategies = [
            'FixedSizeChunker',
            'SemanticChunker', 
            'StructureAwareChunker',
            'HybridChunker'
        ]
        
        self.embedding_models = [
            'SentenceTransformersEmbeddings',
            'BGEEmbeddings',
            'E5Embeddings',
            'SciBERTEmbeddings',
            'LayoutLMEmbeddings',
            'OpenAIEmbeddings'
        ]
        
        self.vector_databases = [
            'FAISSVectorDB',
            'ChromaVectorDB',
            'MilvusVectorDB',
            'QdrantVectorDB',
            'WeaviateVectorDB',
            'PineconeVectorDB'
        ]
        
        # Generate ALL possible combinations
        self.pipeline_combinations = self._generate_all_combinations()
        
        logger.info(f"Generated {len(self.pipeline_combinations)} pipeline combinations")
        logger.info(f"Combinations: {len(self.chunking_strategies)} chunking × {len(self.embedding_models)} embeddings × {len(self.vector_databases)} vector DBs = {len(self.pipeline_combinations)} total")
    
    def _generate_all_combinations(self):
        """Generate all possible combinations of components."""
        combinations = []
        combination_id = 1
        
        for chunking in self.chunking_strategies:
            for embedding in self.embedding_models:
                for vectordb in self.vector_databases:
                    
                    # Create readable name
                    chunking_short = chunking.replace('Chunker', '').replace('Embeddings', '')
                    embedding_short = embedding.replace('Embeddings', '').replace('Transformers', 'Trans')
                    vectordb_short = vectordb.replace('VectorDB', '')
                    
                    name = f"{chunking_short}_{embedding_short}_{vectordb_short}"
                    
                    # Create description
                    chunking_desc = {
                        'FixedSizeChunker': 'Fixed-size chunking',
                        'SemanticChunker': 'Semantic boundary chunking',
                        'StructureAwareChunker': 'Structure-preserving chunking', 
                        'HybridChunker': 'Hybrid chunking strategy'
                    }
                    
                    embedding_desc = {
                        'SentenceTransformersEmbeddings': 'Sentence Transformers',
                        'BGEEmbeddings': 'BGE embeddings',
                        'E5Embeddings': 'E5 embeddings',
                        'SciBERTEmbeddings': 'SciBERT embeddings',
                        'LayoutLMEmbeddings': 'LayoutLM embeddings',
                        'OpenAIEmbeddings': 'OpenAI embeddings'
                    }
                    
                    vectordb_desc = {
                        'FAISSVectorDB': 'FAISS vector store',
                        'ChromaVectorDB': 'ChromaDB vector store',
                        'MilvusVectorDB': 'Milvus vector store',
                        'QdrantVectorDB': 'Qdrant vector store',
                        'WeaviateVectorDB': 'Weaviate vector store',
                        'PineconeVectorDB': 'Pinecone vector store'
                    }
                    
                    description = f"{chunking_desc[chunking]} + {embedding_desc[embedding]} + {vectordb_desc[vectordb]}"
                    
                    combinations.append({
                        'id': combination_id,
                        'name': name,
                        'chunking': chunking,
                        'embeddings': embedding,
                        'vectordb': vectordb,
                        'description': description
                    })
                    
                    combination_id += 1
        
        return combinations
    
    def load_test_questions(self) -> List[Dict]:
        """Load test questions from JSON file."""
        try:
            with open('data/test_questions.json', 'r') as f:
                questions = json.load(f)
            logger.info(f"Loaded {len(questions)} test questions")
            return questions
        except FileNotFoundError:
            logger.error("Test questions file not found!")
            return []
    
    def setup_rag_pipeline(self, config: Dict) -> Dict:
        """Setup a RAG pipeline with specified configuration."""
        try:
            components = {}
            
            # Initialize chunking
            if config['chunking'] == 'FixedSizeChunker':
                from chunking import FixedSizeChunker
                components['chunker'] = FixedSizeChunker(chunk_size=512, overlap=50)
            elif config['chunking'] == 'SemanticChunker':
                from chunking import SemanticChunker  
                components['chunker'] = SemanticChunker(similarity_threshold=0.8)
            elif config['chunking'] == 'StructureAwareChunker':
                from chunking import StructureAwareChunker
                components['chunker'] = StructureAwareChunker(preserve_headers=True)
            elif config['chunking'] == 'HybridChunker':
                from chunking import HybridChunker
                components['chunker'] = HybridChunker()
            
            # Initialize embeddings
            if config['embeddings'] == 'SentenceTransformersEmbeddings':
                from embeddings import SentenceTransformersEmbeddings
                components['embeddings'] = SentenceTransformersEmbeddings()
            elif config['embeddings'] == 'BGEEmbeddings':
                from embeddings import BGEEmbeddings
                components['embeddings'] = BGEEmbeddings()
            elif config['embeddings'] == 'E5Embeddings':
                from embeddings import E5Embeddings
                components['embeddings'] = E5Embeddings()
            elif config['embeddings'] == 'SciBERTEmbeddings':
                from embeddings import SciBERTEmbeddings
                components['embeddings'] = SciBERTEmbeddings()
            elif config['embeddings'] == 'LayoutLMEmbeddings':
                from embeddings import LayoutLMEmbeddings
                components['embeddings'] = LayoutLMEmbeddings()
            elif config['embeddings'] == 'OpenAIEmbeddings':
                from embeddings import OpenAIEmbeddings
                components['embeddings'] = OpenAIEmbeddings()
            
            # Initialize vector database
            if config['vectordb'] == 'FAISSVectorDB':
                from vectordb import FAISSVectorDB
                components['vectordb'] = FAISSVectorDB(components['embeddings'])
            elif config['vectordb'] == 'ChromaVectorDB':
                from vectordb import ChromaVectorDB
                components['vectordb'] = ChromaVectorDB(components['embeddings'], 
                                                      collection_name=f"test_{config['name']}")
            elif config['vectordb'] == 'MilvusVectorDB':
                from vectordb import MilvusVectorDB
                components['vectordb'] = MilvusVectorDB(components['embeddings'],
                                                       collection_name=f"test_{config['name']}")
            elif config['vectordb'] == 'QdrantVectorDB':
                from vectordb import QdrantVectorDB
                components['vectordb'] = QdrantVectorDB(components['embeddings'],
                                                       collection_name=f"test_{config['name']}")
            elif config['vectordb'] == 'WeaviateVectorDB':
                from vectordb import WeaviateVectorDB
                components['vectordb'] = WeaviateVectorDB(components['embeddings'],
                                                         index_name=f"Test{config['name']}")
            elif config['vectordb'] == 'PineconeVectorDB':
                from vectordb import PineconeVectorDB
                components['vectordb'] = PineconeVectorDB(components['embeddings'],
                                                         index_name=f"test-{config['name'].lower()}")
            
            logger.info(f"✓ Successfully setup pipeline: {config['name']}")
            return components
            
        except Exception as e:
            logger.error(f"✗ Failed to setup pipeline {config['name']}: {e}")
            return None
    
    def test_pipeline_with_question(self, pipeline_config: Dict, question: Dict, 
                                  document_text: str) -> Dict:
        """Test a single pipeline with a single question."""
        try:
            # Setup pipeline
            components = self.setup_rag_pipeline(pipeline_config)
            if not components:
                return {
                    'pipeline': pipeline_config['name'],
                    'question_id': question['id'],
                    'success': False,
                    'error': 'Pipeline setup failed',
                    'score': 0
                }
            
            # Process document
            chunks = components['chunker'].chunk(document_text)
            logger.debug(f"Generated {len(chunks)} chunks")
            
            # Add to vector store (limit chunks for testing performance)
            max_chunks = min(50, len(chunks))  # Increased from 20 to 50 for better coverage
            chunk_ids = components['vectordb'].add_texts(chunks[:max_chunks])
            logger.debug(f"Added {len(chunk_ids)} chunks to vector store")
            
            # Retrieve relevant chunks
            relevant_chunks = components['vectordb'].similarity_search(question['question'], k=5)
            logger.debug(f"Retrieved {len(relevant_chunks)} relevant chunks")
            
            # Prepare context
            context = "\\n\\n".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) 
                                   for doc in relevant_chunks])
            
            # Generate response
            prompt = f"""Based on the following technical documentation context, please answer the question accurately and completely.

Context:
{context}

Question: {question['question']}

Please provide a precise, technical answer based only on the information provided in the context."""
            
            response = self.llm.generate_with_context(context, question['question'])
            
            # Score the response
            scores = self.scorer.score_response(question, response)
            
            return {
                'pipeline': pipeline_config['name'],
                'question_id': question['id'],
                'question': question['question'],
                'response': response,
                'expected_answer': question['expected_answer'],
                'success': True,
                'scores': scores,
                'total_score': scores['total_score'],
                'retrieval_context_length': len(context),
                'num_chunks_retrieved': len(relevant_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error testing pipeline {pipeline_config['name']} with question {question['id']}: {e}")
            return {
                'pipeline': pipeline_config['name'],
                'question_id': question['id'],
                'success': False,
                'error': str(e),
                'total_score': 0
            }
    
    def run_comprehensive_test(self) -> pd.DataFrame:
        """Run comprehensive test across all pipeline combinations."""
        
        # Load test questions
        questions = self.load_test_questions()
        if not questions:
            logger.error("No questions loaded!")
            return pd.DataFrame()
        
        # Load document
        document_text = self._load_test_document()
        if not document_text:
            logger.error("No document loaded!")
            return pd.DataFrame()
        
        logger.info(f"Starting comprehensive test with {len(self.pipeline_combinations)} pipeline combinations")
        logger.info(f"Testing {len(questions)} questions per pipeline")
        
        all_results = []
        total_tests = len(self.pipeline_combinations) * len(questions)
        current_test = 0
        
        for pipeline_config in self.pipeline_combinations:
            logger.info(f"\\n{'='*60}")
            logger.info(f"Testing Pipeline: {pipeline_config['name']}")
            logger.info(f"Description: {pipeline_config['description']}")
            logger.info(f"{'='*60}")
            
            pipeline_results = []
            
            for question in questions:
                current_test += 1
                progress = (current_test / total_tests) * 100
                
                logger.info(f"[{progress:.1f}%] Testing Q{question['id']} with {pipeline_config['name']}")
                
                result = self.test_pipeline_with_question(pipeline_config, question, document_text)
                result['pipeline_description'] = pipeline_config['description']
                pipeline_results.append(result)
                all_results.append(result)
                
                if result['success']:
                    logger.info(f"  ✓ Score: {result['total_score']:.2f}/10")
                else:
                    logger.error(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
                
                # Small delay to prevent API rate limits
                time.sleep(0.5)
            
            # Calculate pipeline summary
            successful_tests = [r for r in pipeline_results if r['success']]
            if successful_tests:
                avg_score = sum(r['total_score'] for r in successful_tests) / len(successful_tests)
                logger.info(f"\\nPipeline {pipeline_config['name']} Average Score: {avg_score:.2f}/10")
                logger.info(f"Success Rate: {len(successful_tests)}/{len(questions)} ({len(successful_tests)/len(questions)*100:.1f}%)")
        
        # Create results DataFrame
        df = pd.DataFrame(all_results)
        
        return df
    
    def _load_test_document(self) -> str:
        """Load test document."""
        try:
            # Try to load from configured path first
            if os.path.exists(TEST_DOCUMENT_PATH):
                with open(TEST_DOCUMENT_PATH, 'r', encoding='utf-8') as f:
                    return f.read()
            
            # Fallback to sample document
            sample_doc = """
            Technical Drilling and Completion Specifications
            
            Casing and Tubing Specifications:
            - 10.75" Casing in 13" hole: Gallons per linear ft: 2.1802, Linear ft per gallon: 0.4587
            - 11.75" casing in 14.5" hole: Cu. Ft. per Linear Ft: 0.3937
            
            Coiled Tubing Specifications:
            For 1.0" diameter coiled tubing with 0.080" wall thickness:
            - 70 kpsi yield strength: 2% ovality no tensile: 4,770 psi, with tensile: 3,387 psi
            - 80 kpsi yield strength: 2% ovality no tensile: 5,593 psi, with tensile: 3,901 psi  
            - 100 kpsi yield strength: 2% ovality no tensile: 6,633 psi, with tensile: 4,727 psi
            
            Safety Factors:
            - Maximum Allowable Safety Factor (SF) for tubing utilization between 50-60%: 0.72
            
            PDC Coring Bits:
            - FCI274LI: Recommended WOB: 250-3,500 lb/in. diameter
            
            Fluid Recommendations:
            - BARO-LUBE GOLD SEAL: Recommended for polymer water-based fluids
            
            Cementing Materials:
            - Micro Fly Ash: Bulk Weight: 1041 Kg/m³ (65 lbs/cu ft)
            - Diacel A: Bulk Weight: 60.3 lbs/cu ft, Specific Gravity: 2.62, 
              Absolute: 0.0458 gals/lb, Volume: 0.0061 cu ft/lb, Activity: 100%, 
              State: Dry, Water Requirements: none
            
            Testing Data:
            - 0% bentonite in 1220m API casing test: Thickening Time: 3:00+ (3 hours or more)
            - 94 pound cement compressive strength:
              * 0 pounds sand: 95°F/800psi: 2,085 psi, 110°F/1600psi: 2,925 psi
              * 50 pounds sand: 95°F/800psi: 2,036 psi, 110°F/1600psi: 3,385 psi
            """
            
            logger.warning("Using sample document for testing")
            return sample_doc
            
        except Exception as e:
            logger.error(f"Failed to load test document: {e}")
            return ""
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """Generate comprehensive test report."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate summary table
        summary_data = []
        for pipeline_config in self.pipeline_combinations:
            pipeline_results = results_df[results_df['pipeline'] == pipeline_config['name']]
            successful_results = pipeline_results[pipeline_results['success'] == True]
            
            if len(successful_results) > 0:
                avg_score = successful_results['total_score'].mean()
                max_score = successful_results['total_score'].max()
                min_score = successful_results['total_score'].min()
                success_rate = len(successful_results) / len(pipeline_results) * 100
            else:
                avg_score = max_score = min_score = success_rate = 0
            
            summary_data.append({
                'Rank': 0,  # Will be filled after sorting
                'Pipeline': pipeline_config['name'],
                'Chunking': pipeline_config['chunking'].replace('Chunker', ''),
                'Embeddings': pipeline_config['embeddings'].replace('Embeddings', ''),
                'Vector DB': pipeline_config['vectordb'].replace('VectorDB', ''),
                'Avg Score': f"{avg_score:.2f}/10",
                'Max Score': f"{max_score:.2f}/10", 
                'Min Score': f"{min_score:.2f}/10",
                'Success Rate': f"{success_rate:.1f}%",
                'Tests': f"{len(successful_results)}/{len(pipeline_results)}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by average score and add rank
        summary_df['_avg_score_num'] = summary_df['Avg Score'].str.extract('(\\d+\\.\\d+)').astype(float)
        summary_df = summary_df.sort_values('_avg_score_num', ascending=False).drop('_avg_score_num', axis=1).reset_index(drop=True)
        summary_df['Rank'] = range(1, len(summary_df) + 1)
        
        # Save comprehensive CSV report
        detailed_output_file = f"{OUTPUT_DIR}/comprehensive_rag_test_detailed_{timestamp}.csv"
        results_df.to_csv(detailed_output_file, index=False)
        
        # Save summary CSV
        summary_output_file = f"{OUTPUT_DIR}/comprehensive_rag_test_summary_{timestamp}.csv"
        summary_df.to_csv(summary_output_file, index=False)
        
        # Generate markdown report
        markdown_report = f"""# Comprehensive RAG Pipeline Test Report - ALL COMBINATIONS
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report evaluates **ALL {len(self.pipeline_combinations)} possible pipeline configurations** across {len(results_df['question_id'].unique())} technical questions.

### Component Matrix Tested
- **Chunking Strategies**: {len(self.chunking_strategies)} options ({', '.join([c.replace('Chunker', '') for c in self.chunking_strategies])})
- **Embedding Models**: {len(self.embedding_models)} options ({', '.join([e.replace('Embeddings', '') for e in self.embedding_models])})
- **Vector Databases**: {len(self.vector_databases)} options ({', '.join([v.replace('VectorDB', '') for v in self.vector_databases])})

**Total Combinations**: {len(self.chunking_strategies)} × {len(self.embedding_models)} × {len(self.vector_databases)} = {len(self.pipeline_combinations)} pipelines
**Total Tests**: {len(results_df)} individual tests

## Top 10 Pipeline Performance

{summary_df.head(10).to_markdown(index=False)}

## Performance Analysis

### Overall Statistics
- **Best Average Score**: {summary_df.iloc[0]['Avg Score']} ({summary_df.iloc[0]['Pipeline']})
- **Highest Single Score**: {summary_df['Max Score'].iloc[0]} 
- **Overall Success Rate**: {results_df['success'].sum()}/{len(results_df)} ({results_df['success'].mean()*100:.1f}%)
- **Average Score Across All Pipelines**: {results_df[results_df['success']==True]['total_score'].mean():.2f}/10

### Component Analysis"""
        
        # Add component-wise analysis
        successful_results = results_df[results_df['success'] == True].copy()
        if len(successful_results) > 0:
            # Analyze by chunking strategy
            markdown_report += "\\n\\n#### Best Chunking Strategies (by average score)\\n"
            chunking_analysis = []
            for strategy in self.chunking_strategies:
                strategy_results = successful_results[successful_results['pipeline'].str.contains(strategy.replace('Chunker', ''))]
                if len(strategy_results) > 0:
                    avg_score = strategy_results['total_score'].mean()
                    chunking_analysis.append({'Strategy': strategy.replace('Chunker', ''), 'Avg Score': f"{avg_score:.2f}", 'Tests': len(strategy_results)})
            
            chunking_df = pd.DataFrame(chunking_analysis).sort_values('Avg Score', ascending=False)
            markdown_report += chunking_df.to_markdown(index=False)
            
            # Analyze by embedding model
            markdown_report += "\\n\\n#### Best Embedding Models (by average score)\\n"
            embedding_analysis = []
            for model in self.embedding_models:
                model_results = successful_results[successful_results['pipeline'].str.contains(model.replace('Embeddings', ''))]
                if len(model_results) > 0:
                    avg_score = model_results['total_score'].mean()
                    embedding_analysis.append({'Model': model.replace('Embeddings', ''), 'Avg Score': f"{avg_score:.2f}", 'Tests': len(model_results)})
            
            embedding_df = pd.DataFrame(embedding_analysis).sort_values('Avg Score', ascending=False)
            markdown_report += embedding_df.to_markdown(index=False)
            
            # Analyze by vector database
            markdown_report += "\\n\\n#### Vector Database Performance\\n"
            vectordb_analysis = []
            for vectordb in self.vector_databases:
                vectordb_results = successful_results[successful_results['pipeline'].str.contains(vectordb.replace('VectorDB', ''))]
                if len(vectordb_results) > 0:
                    avg_score = vectordb_results['total_score'].mean()
                    vectordb_analysis.append({'Vector DB': vectordb.replace('VectorDB', ''), 'Avg Score': f"{avg_score:.2f}", 'Tests': len(vectordb_results)})
            
            vectordb_df = pd.DataFrame(vectordb_analysis).sort_values('Avg Score', ascending=False)
            markdown_report += vectordb_df.to_markdown(index=False)
        
        # Add per-question analysis
        for question_id in results_df['question_id'].unique():
            question_results = results_df[results_df['question_id'] == question_id]
            successful_results = question_results[question_results['success'] == True]
            
            if len(successful_results) > 0:
                best_result = successful_results.loc[successful_results['total_score'].idxmax()]
                markdown_report += f"""
### {question_id}
- **Question**: {best_result['question']}
- **Best Pipeline**: {best_result['pipeline']} (Score: {best_result['total_score']:.2f}/10)
- **Average Score**: {successful_results['total_score'].mean():.2f}/10
"""
        
        markdown_report += f"""

## Files Generated
- Detailed Results: `{detailed_output_file}`
- Summary Table: `{summary_output_file}`
- This Report: `comprehensive_rag_test_report_{timestamp}.md`

## Methodology
- **Scoring System**: Advanced 4-component scoring (keywords 40%, numerical 30%, semantic 20%, completeness 10%)
- **Vector Retrieval**: Top-5 chunks per question
- **Context Limit**: Maximum 20 chunks per pipeline to ensure consistent performance
- **LLM**: Gemini Pro for response generation
"""
        
        # Save markdown report
        markdown_file = f"{OUTPUT_DIR}/comprehensive_rag_test_report_{timestamp}.md"
        with open(markdown_file, 'w') as f:
            f.write(markdown_report)
        
        # Print summary to console
        print("\\n" + "="*80)
        print("🎯 COMPREHENSIVE RAG PIPELINE TEST RESULTS")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("\\n" + "="*80)
        print(f"📊 Reports saved:")
        print(f"  • Detailed: {detailed_output_file}")
        print(f"  • Summary: {summary_output_file}")
        print(f"  • Report: {markdown_file}")
        print("="*80)
        
        return summary_df

def main():
    """Run comprehensive RAG testing."""
    
    # Validate configuration
    if not validate_config():
        logger.error("Configuration validation failed!")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize tester
    tester = ComprehensiveRAGTester()
    
    # Run comprehensive test
    logger.info("🚀 Starting Comprehensive RAG Pipeline Testing...")
    results_df = tester.run_comprehensive_test()
    
    if results_df.empty:
        logger.error("No results generated!")
        return
    
    # Generate comprehensive report
    logger.info("📊 Generating comprehensive report...")
    summary_df = tester.generate_comprehensive_report(results_df)
    
    logger.info("✅ Comprehensive testing completed successfully!")

if __name__ == "__main__":
    main()