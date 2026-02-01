"""Concrete metrics for RAG pipeline comparison."""

import numpy as np
from typing import List, Dict, Any, Tuple
import time
import re
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


class RAGComparisonMetrics:
    """Concrete metrics for comparing RAG pipeline configurations."""
    
    def __init__(self, embeddings_model=None):
        self.embeddings_model = embeddings_model
        
    def calculate_retrieval_metrics(self, query: str, retrieved_chunks: List[Dict], 
                                  ground_truth_chunks: List[str]) -> Dict[str, float]:
        """Calculate retrieval quality metrics."""
        retrieved_texts = [chunk.get('text', str(chunk)) for chunk in retrieved_chunks]
        
        # Precision@K (top 5)
        k = 5
        top_k = retrieved_texts[:k]
        relevant_in_top_k = sum(1 for chunk in top_k if self._is_relevant(chunk, ground_truth_chunks))
        precision_at_5 = relevant_in_top_k / k if k > 0 else 0
        
        # Recall@K  
        total_relevant_found = sum(1 for chunk in top_k if self._is_relevant(chunk, ground_truth_chunks))
        recall_at_5 = total_relevant_found / len(ground_truth_chunks) if ground_truth_chunks else 0
        
        # Mean Reciprocal Rank
        mrr = 0
        for i, chunk in enumerate(retrieved_texts):
            if self._is_relevant(chunk, ground_truth_chunks):
                mrr = 1 / (i + 1)
                break
        
        # Average retrieval score
        avg_retrieval_score = np.mean([chunk.get('score', 0) for chunk in retrieved_chunks])
        
        return {
            'precision_at_5': precision_at_5,
            'recall_at_5': recall_at_5,
            'mrr': mrr,
            'avg_retrieval_score': avg_retrieval_score
        }
    
    def calculate_answer_quality_metrics(self, predicted_answer: str, 
                                       ground_truth: str, 
                                       source_context: str) -> Dict[str, float]:
        """Calculate answer quality metrics."""
        
        # 1. Token-level F1 Score
        f1_score = self._calculate_f1_score(predicted_answer, ground_truth)
        
        # 2. BLEU Score (n-gram overlap)
        bleu_score = self._calculate_bleu_score(predicted_answer, ground_truth)
        
        # 3. Semantic Similarity (if embeddings available)
        semantic_sim = 0.0
        if self.embeddings_model:
            pred_emb = self.embeddings_model.embed_text(predicted_answer)
            truth_emb = self.embeddings_model.embed_text(ground_truth)
            semantic_sim = cosine_similarity([pred_emb], [truth_emb])[0][0]
        
        # 4. Answer Completeness (concept coverage)
        completeness = self._calculate_completeness(predicted_answer, ground_truth)
        
        # 5. Context Adherence (how much answer is grounded in context)
        context_adherence = self._calculate_context_adherence(predicted_answer, source_context)
        
        return {
            'f1_score': f1_score,
            'bleu_score': bleu_score,
            'semantic_similarity': semantic_sim,
            'answer_completeness': completeness,
            'context_adherence': context_adherence
        }
    
    def calculate_factual_accuracy_metrics(self, predicted_answer: str, 
                                         source_context: str) -> Dict[str, float]:
        """Calculate factual accuracy metrics."""
        
        # 1. Entity Accuracy
        entity_accuracy = self._calculate_entity_accuracy(predicted_answer, source_context)
        
        # 2. Numerical Accuracy  
        numerical_accuracy = self._calculate_numerical_accuracy(predicted_answer, source_context)
        
        # 3. Factual Consistency Score
        factual_consistency = self._calculate_factual_consistency(predicted_answer, source_context)
        
        # 4. Hallucination Risk Score (0-1, lower is better)
        hallucination_risk = self._calculate_hallucination_risk(predicted_answer, source_context)
        
        return {
            'entity_accuracy': entity_accuracy,
            'numerical_accuracy': numerical_accuracy,
            'factual_consistency': factual_consistency,
            'hallucination_risk': hallucination_risk
        }
    
    def calculate_performance_metrics(self, start_time: float, end_time: float,
                                    memory_before: float, memory_after: float,
                                    num_tokens: int) -> Dict[str, float]:
        """Calculate performance metrics."""
        
        response_time = end_time - start_time
        memory_usage = memory_after - memory_before
        tokens_per_second = num_tokens / response_time if response_time > 0 else 0
        
        return {
            'response_time': response_time,
            'memory_usage_mb': memory_usage,
            'tokens_per_second': tokens_per_second,
            'efficiency_score': tokens_per_second / memory_usage if memory_usage > 0 else 0
        }
    
    def calculate_composite_score(self, retrieval_metrics: Dict, quality_metrics: Dict,
                                accuracy_metrics: Dict, performance_metrics: Dict,
                                weights: Dict = None) -> Dict[str, float]:
        """Calculate weighted composite score for direct comparison."""
        
        # Default weights (can be customized based on use case)
        default_weights = {
            'retrieval_weight': 0.25,
            'quality_weight': 0.35, 
            'accuracy_weight': 0.30,
            'performance_weight': 0.10
        }
        weights = weights or default_weights
        
        # Normalize metrics to 0-1 scale
        retrieval_score = (
            retrieval_metrics['precision_at_5'] * 0.4 +
            retrieval_metrics['recall_at_5'] * 0.3 +
            retrieval_metrics['mrr'] * 0.3
        )
        
        quality_score = (
            quality_metrics['f1_score'] * 0.3 +
            quality_metrics['semantic_similarity'] * 0.3 +
            quality_metrics['answer_completeness'] * 0.2 +
            quality_metrics['context_adherence'] * 0.2
        )
        
        accuracy_score = (
            accuracy_metrics['entity_accuracy'] * 0.3 +
            accuracy_metrics['numerical_accuracy'] * 0.3 +
            accuracy_metrics['factual_consistency'] * 0.2 +
            (1 - accuracy_metrics['hallucination_risk']) * 0.2  # Invert hallucination risk
        )
        
        # Normalize performance metrics (lower time/memory is better)
        max_response_time = 10.0  # Assume 10s is worst case
        max_memory = 1000.0  # Assume 1GB is worst case
        performance_score = (
            max(0, 1 - (performance_metrics['response_time'] / max_response_time)) * 0.6 +
            max(0, 1 - (performance_metrics['memory_usage_mb'] / max_memory)) * 0.4
        )
        
        # Calculate weighted composite score
        composite_score = (
            retrieval_score * weights['retrieval_weight'] +
            quality_score * weights['quality_weight'] +
            accuracy_score * weights['accuracy_weight'] +
            performance_score * weights['performance_weight']
        )
        
        return {
            'retrieval_score': retrieval_score,
            'quality_score': quality_score, 
            'accuracy_score': accuracy_score,
            'performance_score': performance_score,
            'composite_score': composite_score,
            'grade': self._score_to_grade(composite_score)
        }
    
    # Helper methods
    def _is_relevant(self, retrieved_chunk: str, ground_truth_chunks: List[str]) -> bool:
        """Check if retrieved chunk is relevant (basic overlap check)."""
        retrieved_tokens = set(retrieved_chunk.lower().split())
        for truth_chunk in ground_truth_chunks:
            truth_tokens = set(truth_chunk.lower().split())
            overlap = len(retrieved_tokens & truth_tokens) / len(truth_tokens) if truth_tokens else 0
            if overlap > 0.3:  # 30% overlap threshold
                return True
        return False
    
    def _calculate_f1_score(self, predicted: str, ground_truth: str) -> float:
        """Calculate token-level F1 score."""
        pred_tokens = set(predicted.lower().split())
        truth_tokens = set(ground_truth.lower().split())
        
        if not pred_tokens or not truth_tokens:
            return 0.0
            
        intersection = pred_tokens & truth_tokens
        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(truth_tokens)
        
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def _calculate_bleu_score(self, predicted: str, ground_truth: str) -> float:
        """Calculate BLEU score (simplified)."""
        pred_tokens = predicted.lower().split()
        truth_tokens = ground_truth.lower().split()
        
        if not pred_tokens:
            return 0.0
        
        # 1-gram precision
        pred_counter = Counter(pred_tokens)
        truth_counter = Counter(truth_tokens)
        
        matches = sum((pred_counter & truth_counter).values())
        precision = matches / len(pred_tokens)
        
        # Brevity penalty
        bp = 1.0 if len(pred_tokens) >= len(truth_tokens) else np.exp(1 - len(truth_tokens) / len(pred_tokens))
        
        return bp * precision
    
    def _calculate_completeness(self, predicted: str, ground_truth: str) -> float:
        """Calculate answer completeness."""
        # Extract key concepts (nouns, proper nouns, numbers)
        pred_concepts = set(re.findall(r'\b[A-Z][a-zA-Z]+\b|\b\d+\.?\d*\b|\b[a-z]{4,}\b', predicted))
        truth_concepts = set(re.findall(r'\b[A-Z][a-zA-Z]+\b|\b\d+\.?\d*\b|\b[a-z]{4,}\b', ground_truth))
        
        if not truth_concepts:
            return 1.0
            
        covered_concepts = pred_concepts & truth_concepts
        return len(covered_concepts) / len(truth_concepts)
    
    def _calculate_context_adherence(self, predicted: str, context: str) -> float:
        """Calculate how well answer adheres to source context."""
        pred_tokens = set(predicted.lower().split())
        context_tokens = set(context.lower().split())
        
        if not pred_tokens:
            return 1.0
            
        grounded_tokens = pred_tokens & context_tokens
        return len(grounded_tokens) / len(pred_tokens)
    
    def _calculate_entity_accuracy(self, predicted: str, context: str) -> float:
        """Calculate entity accuracy."""
        # Simple entity extraction (proper nouns)
        pred_entities = set(re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', predicted))
        context_entities = set(re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', context))
        
        if not pred_entities:
            return 1.0
            
        accurate_entities = pred_entities & context_entities
        return len(accurate_entities) / len(pred_entities)
    
    def _calculate_numerical_accuracy(self, predicted: str, context: str) -> float:
        """Calculate numerical accuracy."""
        pred_numbers = set(re.findall(r'\b\d+\.?\d*\b', predicted))
        context_numbers = set(re.findall(r'\b\d+\.?\d*\b', context))
        
        if not pred_numbers:
            return 1.0
            
        accurate_numbers = pred_numbers & context_numbers
        return len(accurate_numbers) / len(pred_numbers)
    
    def _calculate_factual_consistency(self, predicted: str, context: str) -> float:
        """Calculate overall factual consistency."""
        entity_acc = self._calculate_entity_accuracy(predicted, context)
        numerical_acc = self._calculate_numerical_accuracy(predicted, context)
        context_adh = self._calculate_context_adherence(predicted, context)
        
        return (entity_acc + numerical_acc + context_adh) / 3
    
    def _calculate_hallucination_risk(self, predicted: str, context: str) -> float:
        """Calculate hallucination risk score."""
        # Check for specific hallucination indicators
        risk_indicators = 0
        total_checks = 4
        
        # 1. Unsupported entities
        if self._calculate_entity_accuracy(predicted, context) < 0.8:
            risk_indicators += 1
            
        # 2. Unsupported numbers
        if self._calculate_numerical_accuracy(predicted, context) < 0.8:
            risk_indicators += 1
            
        # 3. Low context adherence
        if self._calculate_context_adherence(predicted, context) < 0.4:
            risk_indicators += 1
            
        # 4. Definitive claims not in context
        definitive_phrases = ['is exactly', 'always', 'never', 'definitely', 'certainly']
        if any(phrase in predicted.lower() for phrase in definitive_phrases):
            if self._calculate_context_adherence(predicted, context) < 0.6:
                risk_indicators += 1
        
        return risk_indicators / total_checks
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade."""
        if score >= 0.9:
            return 'A+'
        elif score >= 0.85:
            return 'A'
        elif score >= 0.8:
            return 'A-'
        elif score >= 0.75:
            return 'B+'
        elif score >= 0.7:
            return 'B'
        elif score >= 0.65:
            return 'B-'
        elif score >= 0.6:
            return 'C+'
        elif score >= 0.55:
            return 'C'
        elif score >= 0.5:
            return 'C-'
        else:
            return 'D'


# Usage example for direct comparison
def compare_two_configurations(config_a_results: Dict, config_b_results: Dict) -> Dict[str, Any]:
    """Direct comparison between two configurations."""
    
    comparison = {
        'winner_by_metric': {},
        'score_differences': {},
        'overall_winner': None,
        'confidence': 0.0
    }
    
    metrics_to_compare = ['composite_score', 'retrieval_score', 'quality_score', 'accuracy_score', 'performance_score']
    
    config_a_wins = 0
    config_b_wins = 0
    
    for metric in metrics_to_compare:
        score_a = config_a_results.get(metric, 0)
        score_b = config_b_results.get(metric, 0)
        
        if score_a > score_b:
            comparison['winner_by_metric'][metric] = 'Config A'
            config_a_wins += 1
        elif score_b > score_a:
            comparison['winner_by_metric'][metric] = 'Config B'  
            config_b_wins += 1
        else:
            comparison['winner_by_metric'][metric] = 'Tie'
            
        comparison['score_differences'][metric] = abs(score_a - score_b)
    
    # Overall winner
    if config_a_wins > config_b_wins:
        comparison['overall_winner'] = 'Config A'
        comparison['confidence'] = config_a_wins / len(metrics_to_compare)
    elif config_b_wins > config_a_wins:
        comparison['overall_winner'] = 'Config B'
        comparison['confidence'] = config_b_wins / len(metrics_to_compare)
    else:
        comparison['overall_winner'] = 'Tie'
        comparison['confidence'] = 0.5
    
    return comparison