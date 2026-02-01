"""Hallucination detection and evaluation."""

from typing import List, Dict, Any, Set
import re
from collections import Counter


class HallucinationDetector:
    """Detect and evaluate hallucinations in RAG responses."""
    
    def __init__(self):
        self.factual_keywords = {
            'temporal': ['yesterday', 'today', 'tomorrow', 'last year', 'next month'],
            'numerical': ['million', 'billion', 'thousand', 'percent', '%'],
            'geographical': ['located in', 'capital of', 'border with'],
            'definitional': ['is defined as', 'means that', 'refers to']
        }
    
    def detect_numerical_hallucination(self, response: str, source_context: str) -> Dict[str, Any]:
        """Detect numerical hallucinations by comparing numbers in response vs context.
        
        Args:
            response: Generated response
            source_context: Source context/documents
            
        Returns:
            Dictionary with hallucination detection results
        """
        # Extract numbers from both texts
        response_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', response))
        context_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', source_context))
        
        # Find numbers in response that don't appear in context
        hallucinated_numbers = response_numbers - context_numbers
        
        return {
            'has_numerical_hallucination': len(hallucinated_numbers) > 0,
            'hallucinated_numbers': list(hallucinated_numbers),
            'total_response_numbers': len(response_numbers),
            'supported_numbers': len(response_numbers - hallucinated_numbers)
        }
    
    def detect_factual_hallucination(self, response: str, source_context: str) -> Dict[str, Any]:
        """Detect factual hallucinations using keyword-based approach.
        
        Args:
            response: Generated response
            source_context: Source context/documents
            
        Returns:
            Dictionary with factual hallucination analysis
        """
        response_lower = response.lower()
        context_lower = source_context.lower()
        
        hallucination_signals = []
        
        # Check for temporal claims
        temporal_claims = [kw for kw in self.factual_keywords['temporal'] if kw in response_lower]
        if temporal_claims and not any(kw in context_lower for kw in temporal_claims):
            hallucination_signals.append('temporal_mismatch')
        
        # Check for geographical claims
        geo_claims = [kw for kw in self.factual_keywords['geographical'] if kw in response_lower]
        if geo_claims and not any(kw in context_lower for kw in geo_claims):
            hallucination_signals.append('geographical_mismatch')
        
        # Check for definitional claims
        def_claims = [kw for kw in self.factual_keywords['definitional'] if kw in response_lower]
        if def_claims and not any(kw in context_lower for kw in def_claims):
            hallucination_signals.append('definitional_mismatch')
        
        return {
            'has_factual_hallucination': len(hallucination_signals) > 0,
            'hallucination_types': hallucination_signals,
            'confidence_score': 1.0 - (len(hallucination_signals) * 0.2)
        }
    
    def calculate_context_overlap(self, response: str, source_context: str) -> float:
        """Calculate the overlap between response and source context.
        
        Args:
            response: Generated response
            source_context: Source context/documents
            
        Returns:
            Context overlap ratio (0-1)
        """
        response_tokens = set(response.lower().split())
        context_tokens = set(source_context.lower().split())
        
        if not response_tokens:
            return 0.0
            
        overlap = len(response_tokens & context_tokens)
        return overlap / len(response_tokens)
    
    def detect_entity_hallucination(self, response: str, source_context: str) -> Dict[str, Any]:
        """Detect hallucinated named entities (basic implementation).
        
        Args:
            response: Generated response
            source_context: Source context/documents
            
        Returns:
            Dictionary with entity hallucination analysis
        """
        # Simple regex-based entity extraction (can be improved with NER)
        person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        org_pattern = r'\b[A-Z][A-Z\s&]+(?:Inc|Corp|Ltd|LLC|University|Company)\b'
        
        response_persons = set(re.findall(person_pattern, response))
        response_orgs = set(re.findall(org_pattern, response))
        
        context_persons = set(re.findall(person_pattern, source_context))
        context_orgs = set(re.findall(org_pattern, source_context))
        
        hallucinated_persons = response_persons - context_persons
        hallucinated_orgs = response_orgs - context_orgs
        
        return {
            'hallucinated_persons': list(hallucinated_persons),
            'hallucinated_organizations': list(hallucinated_orgs),
            'has_entity_hallucination': len(hallucinated_persons) > 0 or len(hallucinated_orgs) > 0,
            'total_entities_in_response': len(response_persons) + len(response_orgs)
        }
    
    def comprehensive_hallucination_check(self, response: str, source_context: str) -> Dict[str, Any]:
        """Perform comprehensive hallucination detection.
        
        Args:
            response: Generated response
            source_context: Source context/documents
            
        Returns:
            Comprehensive hallucination analysis
        """
        numerical_analysis = self.detect_numerical_hallucination(response, source_context)
        factual_analysis = self.detect_factual_hallucination(response, source_context)
        entity_analysis = self.detect_entity_hallucination(response, source_context)
        context_overlap = self.calculate_context_overlap(response, source_context)
        
        # Overall hallucination score
        hallucination_indicators = [
            numerical_analysis['has_numerical_hallucination'],
            factual_analysis['has_factual_hallucination'],
            entity_analysis['has_entity_hallucination'],
            context_overlap < 0.3  # Low context overlap indicates potential hallucination
        ]
        
        hallucination_score = sum(hallucination_indicators) / len(hallucination_indicators)
        
        return {
            'numerical_analysis': numerical_analysis,
            'factual_analysis': factual_analysis,
            'entity_analysis': entity_analysis,
            'context_overlap': context_overlap,
            'overall_hallucination_score': hallucination_score,
            'is_likely_hallucinated': hallucination_score > 0.5
        }