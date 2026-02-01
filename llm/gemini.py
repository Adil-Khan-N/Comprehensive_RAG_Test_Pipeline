"""Gemini Pro language model implementation."""

from typing import List, Dict, Any
import google.generativeai as genai
from .base import BaseLLM


class GeminiProLLM(BaseLLM):
    """Gemini Pro language model implementation."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", 
                 temperature: float = 0.7, max_tokens: int = 1000):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
        
        # Generation configuration
        self.generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            candidate_count=1,
        )
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Gemini Pro API.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated text response
        """
        try:
            # Override default config with any provided kwargs
            config = self.generation_config
            if 'temperature' in kwargs:
                config.temperature = kwargs['temperature']
            if 'max_tokens' in kwargs:
                config.max_output_tokens = kwargs['max_tokens']
                
            # Generate response
            response = self.client.generate_content(
                prompt,
                generation_config=config,
                safety_settings={
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
            )
            
            if response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                return "No response generated."
                
        except Exception as e:
            print(f"Error generating response with Gemini Pro: {e}")
            return f"Error: {str(e)}"
        
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional parameters
            
        Returns:
            List of generated text responses
        """
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, **kwargs)
            responses.append(response)
        return responses
    
    def generate_with_context(self, context: str, question: str, **kwargs) -> str:
        """Generate response with structured context and question.
        
        Args:
            context: Retrieved context/documents
            question: User question
            **kwargs: Additional parameters
            
        Returns:
            Generated answer
        """
        prompt = f"""You are an expert assistant analyzing technical documentation. Please provide accurate, concise answers based strictly on the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Answer based only on the information provided in the context
2. If the answer requires specific numerical values, extract them precisely from tables or text
3. If the information is not available in the context, clearly state "The information is not available in the provided context"
4. For table lookups, be extremely precise with values and units
5. Keep the answer focused and direct

Answer:"""
        
        return self.generate(prompt, **kwargs)