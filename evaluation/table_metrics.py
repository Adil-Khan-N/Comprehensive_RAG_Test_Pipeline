"""Table-specific evaluation metrics."""

from typing import List, Dict, Any, Tuple
import re
import pandas as pd


class TableMetrics:
    """Evaluation metrics specific to table understanding and QA."""
    
    @staticmethod
    def table_cell_accuracy(predicted_cells: List[str], ground_truth_cells: List[str]) -> float:
        """Calculate accuracy for table cell extraction.
        
        Args:
            predicted_cells: List of predicted cell values
            ground_truth_cells: List of ground truth cell values
            
        Returns:
            Cell-level accuracy
        """
        if len(predicted_cells) != len(ground_truth_cells):
            return 0.0
            
        correct = sum(1 for p, g in zip(predicted_cells, ground_truth_cells) 
                     if p.strip().lower() == g.strip().lower())
        return correct / len(ground_truth_cells)
    
    @staticmethod
    def table_structure_accuracy(predicted_structure: Dict[str, Any], 
                                ground_truth_structure: Dict[str, Any]) -> float:
        """Calculate accuracy for table structure understanding.
        
        Args:
            predicted_structure: Predicted table structure (rows, cols, headers)
            ground_truth_structure: Ground truth table structure
            
        Returns:
            Structure accuracy score
        """
        scores = []
        
        # Compare number of rows
        if 'rows' in predicted_structure and 'rows' in ground_truth_structure:
            scores.append(1.0 if predicted_structure['rows'] == ground_truth_structure['rows'] else 0.0)
            
        # Compare number of columns
        if 'cols' in predicted_structure and 'cols' in ground_truth_structure:
            scores.append(1.0 if predicted_structure['cols'] == ground_truth_structure['cols'] else 0.0)
            
        # Compare headers
        if 'headers' in predicted_structure and 'headers' in ground_truth_structure:
            pred_headers = set(h.lower().strip() for h in predicted_structure['headers'])
            gt_headers = set(h.lower().strip() for h in ground_truth_structure['headers'])
            header_accuracy = len(pred_headers & gt_headers) / len(gt_headers) if gt_headers else 0.0
            scores.append(header_accuracy)
            
        return sum(scores) / len(scores) if scores else 0.0
    
    @staticmethod
    def numerical_accuracy(predicted_numbers: List[float], 
                          ground_truth_numbers: List[float], 
                          tolerance: float = 1e-3) -> float:
        """Calculate accuracy for numerical values in tables.
        
        Args:
            predicted_numbers: List of predicted numerical values
            ground_truth_numbers: List of ground truth numerical values
            tolerance: Tolerance for floating point comparison
            
        Returns:
            Numerical accuracy score
        """
        if len(predicted_numbers) != len(ground_truth_numbers):
            return 0.0
            
        correct = sum(1 for p, g in zip(predicted_numbers, ground_truth_numbers) 
                     if abs(p - g) <= tolerance)
        return correct / len(ground_truth_numbers)
    
    @staticmethod
    def table_qa_accuracy(predictions: List[str], ground_truths: List[str], 
                         table_contexts: List[str]) -> Dict[str, float]:
        """Evaluate table QA accuracy with table-specific considerations.
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers
            table_contexts: List of table contexts (HTML, CSV, etc.)
            
        Returns:
            Dictionary of table QA metrics
        """
        if len(predictions) != len(ground_truths) or len(predictions) != len(table_contexts):
            raise ValueError("All input lists must have the same length")
        
        exact_matches = []
        numerical_matches = []
        partial_matches = []
        
        for pred, gt, context in zip(predictions, ground_truths, table_contexts):
            # Exact match
            exact_matches.append(1.0 if pred.strip().lower() == gt.strip().lower() else 0.0)
            
            # Try to parse as numbers for numerical comparison
            try:
                pred_num = float(re.sub(r'[^\d.-]', '', pred))
                gt_num = float(re.sub(r'[^\d.-]', '', gt))
                numerical_matches.append(1.0 if abs(pred_num - gt_num) < 1e-3 else 0.0)
            except (ValueError, TypeError):
                numerical_matches.append(0.0)
            
            # Partial match (token overlap)
            pred_tokens = set(pred.lower().split())
            gt_tokens = set(gt.lower().split())
            if gt_tokens:
                partial_match = len(pred_tokens & gt_tokens) / len(gt_tokens)
            else:
                partial_match = 0.0
            partial_matches.append(partial_match)
        
        return {
            'exact_match': sum(exact_matches) / len(exact_matches),
            'numerical_match': sum(numerical_matches) / len(numerical_matches),
            'partial_match': sum(partial_matches) / len(partial_matches)
        }