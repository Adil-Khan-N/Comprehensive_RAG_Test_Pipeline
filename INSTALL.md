# RAG Framework - Installation Guide

## 🚀 Quick Installation

### Option 1: Minimal Installation (Quick Test)
For testing 12 representative pipeline combinations:

```bash
pip install -r requirements_minimal.txt
```

Then run:
```bash
python test_quick_sample.py
```

### Option 2: Full Installation (All 144 Combinations) 
For testing all possible pipeline combinations:

```bash
pip install -r requirements.txt
```

Then run:
```bash
python test_all_combinations.py
```

### Option 3: Manual Installation
If you prefer to install packages individually:

```bash
# Core dependencies
pip install langchain>=0.1.0 langchain-community langchain-text-splitters
pip install sentence-transformers transformers torch
pip install pandas numpy

# Vector databases (choose what you need)
pip install faiss-cpu chromadb
pip install qdrant-client weaviate-client pinecone-client pymilvus  # Optional

# LLM integration  
pip install google-generativeai
pip install openai  # Optional, for OpenAI embeddings

# Environment management
pip install python-dotenv
```

## 🔑 API Keys Required

### Required (for LLM responses)
- **Gemini API**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
  ```bash
  # Add to .env file:
  GEMINI_API_KEY=your_gemini_api_key_here
  ```

### Optional (only if using specific components)
- **OpenAI API**: Only if using OpenAI embeddings
  ```bash
  OPENAI_API_KEY=your_openai_api_key_here
  ```

- **Pinecone API**: Only if using Pinecone vector database
  ```bash
  PINECONE_API_KEY=your_pinecone_api_key_here
  ```

## 🔧 Troubleshooting

### Common Issues

#### `faiss_cpu` not found
The package is `faiss-cpu` but imports as `faiss`:
```bash
pip install faiss-cpu
```

#### GPU vs CPU versions
For GPU acceleration (if you have CUDA):
```bash
pip install faiss-gpu  # Instead of faiss-cpu
```

#### Vector database connection issues
Some vector databases may need additional setup:
- **Milvus**: May require Docker
- **Weaviate**: May require Docker  
- **Qdrant**: Works with cloud or Docker

#### Memory issues
If you run out of memory:
- Use the quick test first: `python test_quick_sample.py`
- Close other applications
- Consider using smaller embedding models

## 📊 Testing Options

| Script | Combinations | Runtime | Use Case |
|--------|-------------|---------|----------|
| `test_quick_sample.py` | 12 | 8-15 min | Initial validation |
| `test_all_combinations.py` | 144 | 45-90 min | Complete analysis |
| `show_components.py` | 0 | <1 min | View available components |

## 🎯 Verification

Test your installation:
```bash
python show_components.py  # View all components
python test_quick_sample.py  # Quick test (recommended first)
```

## 💡 Tips

1. **Start with minimal installation** and quick test
2. **Add components gradually** as needed
3. **Use virtual environment** to avoid conflicts:
   ```bash
   python -m venv rag_env
   rag_env\Scripts\activate  # Windows
   # source rag_env/bin/activate  # Linux/Mac
   ```
4. **Monitor system resources** during full testing
5. **Check logs** in `output/` directory for detailed results