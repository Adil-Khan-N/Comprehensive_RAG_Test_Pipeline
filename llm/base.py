"""Base class for language models."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseLLM(ABC):
    """Abstract base class for language models."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response from prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated text responses
        """
        pass