"""Configuration file for RAG Framework using environment variables."""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# API Keys from environment variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

# Test Configuration
TEST_SCENARIOS = {
    'quick': {
        'max_questions': 5,
        'chunking_strategies': ['simple'],
        'embedding_models': ['sentence_transformer'],
        'vector_dbs': ['faiss'],
        'retrievers': ['similarity'],
        'llms': ['gemini']
    },
    'standard': {
        'max_questions': 15,
        'chunking_strategies': ['simple', 'semantic'],
        'embedding_models': ['sentence_transformer', 'openai'],
        'vector_dbs': ['faiss', 'chroma'],
        'retrievers': ['similarity', 'mmr'],
        'llms': ['gemini']
    },
    'comprehensive': {
        'max_questions': int(os.getenv('MAX_QUESTIONS', 30)),
        'chunking_strategies': ['simple', 'semantic', 'hybrid', 'recursive'],
        'embedding_models': ['sentence_transformer', 'openai', 'cohere'],
        'vector_dbs': ['faiss', 'chroma', 'pinecone'],
        'retrievers': ['similarity', 'mmr', 'ensemble', 'reranking'],
        'llms': ['gemini']
    }
}

# File paths from environment or defaults
TEST_DOCUMENT_PATH = os.getenv('TEST_DOCUMENT_PATH', 'data/marker_md/RedBook_Markdown.md')
TEST_QUESTIONS_PATH = "data/test_questions.json"
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'results')

# Performance settings from environment or defaults
MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', 3))
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 30))

# Validation function
def validate_config():
    """Validate that required configuration is available."""
    missing = []
    
    if not GEMINI_API_KEY or GEMINI_API_KEY == 'your_gemini_api_key_here':
        missing.append('GEMINI_API_KEY')
    
    if not Path(TEST_QUESTIONS_PATH).exists():
        missing.append(f'Test questions file at {TEST_QUESTIONS_PATH}')
    
    if not Path(TEST_DOCUMENT_PATH).exists():
        missing.append(f'Test document at {TEST_DOCUMENT_PATH}')
    
    if missing:
        raise ValueError(f"Missing required configuration: {', '.join(missing)}")
    
    return True