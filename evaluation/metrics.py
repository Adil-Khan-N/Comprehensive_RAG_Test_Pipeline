"""Evaluation metrics for RAG systems."""

from typing import List, Dict, Any
import re
from collections import Counter
import math


class RAGMetrics:
    """Evaluation metrics for RAG systems."""
    
    @staticmethod
    def exact_match(predicted: str, ground_truth: str) -> float:
        """Calculate exact match score.
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            Exact match score (0 or 1)
        """
        return float(predicted.strip().lower() == ground_truth.strip().lower())
    
    @staticmethod
    def f1_score(predicted: str, ground_truth: str) -> float:
        """Calculate F1 score based on token overlap.
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            F1 score
        """
        pred_tokens = set(predicted.lower().split())
        gt_tokens = set(ground_truth.lower().split())
        
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0
            
        intersection = pred_tokens & gt_tokens
        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(gt_tokens)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def bleu_score(predicted: str, ground_truth: str, n: int = 4) -> float:
        """Calculate BLEU score.
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            n: Maximum n-gram order
            
        Returns:
            BLEU score
        """
        def get_ngrams(text: str, n: int) -> Counter:
            tokens = text.lower().split()
            return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
        
        pred_tokens = predicted.lower().split()
        gt_tokens = ground_truth.lower().split()
        
        if len(pred_tokens) == 0:
            return 0.0
            
        # Calculate brevity penalty
        bp = 1.0 if len(pred_tokens) >= len(gt_tokens) else math.exp(1 - len(gt_tokens) / len(pred_tokens))
        
        # Calculate precision for each n-gram order
        precisions = []
        for i in range(1, n + 1):
            pred_ngrams = get_ngrams(predicted, i)
            gt_ngrams = get_ngrams(ground_truth, i)
            
            if len(pred_ngrams) == 0:
                precisions.append(0.0)
                continue
                
            matches = sum((pred_ngrams & gt_ngrams).values())
            precision = matches / len(pred_ngrams)
            precisions.append(precision)
        
        if any(p == 0 for p in precisions):
            return 0.0
            
        # Calculate geometric mean
        geometric_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        return bp * geometric_mean
    
    @staticmethod
    def rouge_l(predicted: str, ground_truth: str) -> float:
        """Calculate ROUGE-L score.
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            ROUGE-L score
        """
        def lcs_length(s1: List[str], s2: List[str]) -> int:
            """Calculate longest common subsequence length."""
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        pred_tokens = predicted.lower().split()
        gt_tokens = ground_truth.lower().split()
        
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0
            
        lcs_len = lcs_length(pred_tokens, gt_tokens)
        precision = lcs_len / len(pred_tokens)
        recall = lcs_len / len(gt_tokens)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)
    
    def evaluate_batch(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """Evaluate a batch of predictions.
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers
            
        Returns:
            Dictionary of metric scores
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions must match number of ground truths")
        
        em_scores = []
        f1_scores = []
        bleu_scores = []
        rouge_scores = []
        
        for pred, gt in zip(predictions, ground_truths):
            em_scores.append(self.exact_match(pred, gt))
            f1_scores.append(self.f1_score(pred, gt))
            bleu_scores.append(self.bleu_score(pred, gt))
            rouge_scores.append(self.rouge_l(pred, gt))
        
        return {
            'exact_match': sum(em_scores) / len(em_scores),
            'f1_score': sum(f1_scores) / len(f1_scores),
            'bleu_score': sum(bleu_scores) / len(bleu_scores),
            'rouge_l': sum(rouge_scores) / len(rouge_scores)
        }