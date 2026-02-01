# RAG Framework

A comprehensive Retrieval-Augmented Generation (RAG) framework for building, evaluating, and comparing different RAG pipelines. This framework supports multiple chunking strategies, embedding models, vector databases, retrieval methods, and language models.

## 🏗️ Architecture

```
rag_framework/
├── data/                    # Data storage
│   ├── raw_pdfs/           # Original PDF documents
│   ├── marker_md/          # Markdown converted documents
│   └── ocr_outputs/        # OCR processed documents
├── chunking/               # Text chunking strategies
├── embeddings/             # Text embedding models
├── vectordb/               # Vector database implementations
├── retriever/              # Document retrieval methods
├── llm/                    # Language model interfaces
├── evaluation/             # Evaluation metrics and datasets
└── experiments/            # Experiment configurations and runner
```

## 🚀 Features

### Chunking Strategies
- **Fixed Size**: Traditional fixed-size chunking with overlap
- **Semantic**: Semantic boundary-aware chunking
- **Structure Aware**: Preserves document structure (headers, paragraphs)
- **Hybrid**: Combines multiple chunking approaches

### Embedding Models
- **OpenAI**: text-embedding-ada-002 and other OpenAI models
- **Sentence Transformers**: Wide variety of open-source models
- **BGE**: BAAI General Embeddings
- **E5**: Microsoft E5 embeddings
- **SciBERT**: Scientific domain-specific embeddings
- **LayoutLM**: Document layout understanding

### Vector Databases
- **FAISS**: Meta's similarity search library
- **ChromaDB**: Open-source embedding database
- **Milvus**: Cloud-native vector database
- **Qdrant**: Vector similarity search engine
- **Weaviate**: Vector search engine with ML models
- **Pinecone**: Managed vector database service

### Retrieval Methods
- **Dense Retrieval**: Semantic similarity using embeddings
- **BM25**: Traditional sparse retrieval
- **Hybrid**: Combines dense and sparse retrieval
- **Reranking**: Cross-encoder based result refinement

### Language Models
- **OpenAI**: GPT-3.5, GPT-4, and other OpenAI models
- **Local Models**: Support for local LLM inference

### Evaluation Metrics
- **Standard QA Metrics**: Exact Match, F1, BLEU, ROUGE-L
- **Table-Specific Metrics**: Cell accuracy, structure accuracy
- **Hallucination Detection**: Numerical, factual, and entity hallucinations

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag_framework

# Install dependencies
pip install -r requirements.txt

# Optional: Install specific dependencies for each component
pip install faiss-cpu chromadb sentence-transformers
pip install openai pinecone-client qdrant-client
```

## 🔧 Quick Start

### 1. Basic RAG Pipeline

```python
from chunking import FixedSizeChunker
from embeddings import OpenAIEmbeddings
from vectordb import FAISSVectorDB
from retriever import DenseRetriever
from llm import OpenAILLM

# Setup components
chunker = FixedSizeChunker(chunk_size=512, overlap=50)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectordb = FAISSVectorDB(dimension=1536)
retriever = DenseRetriever(embeddings, vectordb)
llm = OpenAILLM(model="gpt-3.5-turbo")

# Process documents
texts = ["Your document text here..."]
chunks = chunker.chunk(texts[0])
vectors = embeddings.embed_batch(chunks)
vectordb.add_vectors(vectors, [f"chunk_{i}" for i in range(len(chunks))])

# Query the system
query = "What is the main topic?"
results = retriever.retrieve(query, k=5)
context = "\\n".join([r['text'] for r in results])
response = llm.generate(f"Context: {context}\\nQuestion: {query}")
```

### 2. Running Experiments

Create an experiment configuration:

```yaml
# experiments/configs/my_experiment.yaml
experiment:
  name: "my_rag_experiment"
  description: "Testing semantic chunking with BGE embeddings"

chunking:
  type: "semantic"
  similarity_threshold: 0.8

embeddings:
  type: "bge"
  model_name: "BAAI/bge-large-en-v1.5"

vectordb:
  type: "faiss"

retriever:
  type: "dense"
  k: 5

llm:
  type: "openai"
  model: "gpt-3.5-turbo"

evaluation:
  dataset_path: "data/qa_dataset.json"
```

Run the experiment:

```bash
cd experiments
python run_experiment.py configs/my_experiment.yaml
```

### 3. Evaluation

```python
from evaluation import QADataset, RAGMetrics, HallucinationDetector

# Load evaluation dataset
dataset = QADataset("data/qa_dataset.json")

# Evaluate predictions
metrics = RAGMetrics()
predictions = ["Answer 1", "Answer 2"]
ground_truths = ["Ground truth 1", "Ground truth 2"]
results = metrics.evaluate_batch(predictions, ground_truths)

# Check for hallucinations
detector = HallucinationDetector()
hallucination_analysis = detector.comprehensive_hallucination_check(
    response="Generated answer",
    source_context="Source document content"
)
```

## 📊 Experiment Configuration

The framework uses YAML configuration files for experiments. Key sections:

- **experiment**: Metadata and description
- **chunking**: Chunking strategy and parameters
- **embeddings**: Embedding model configuration
- **vectordb**: Vector database settings
- **retriever**: Retrieval method and parameters
- **llm**: Language model configuration
- **evaluation**: Dataset and metrics specification

## 🔍 Evaluation Metrics

### Standard Metrics
- **Exact Match**: Binary exact string match
- **F1 Score**: Token-level precision and recall
- **BLEU**: N-gram based evaluation
- **ROUGE-L**: Longest common subsequence

### Advanced Metrics
- **Table Metrics**: Structure and cell-level accuracy
- **Hallucination Detection**: Identifies factual inconsistencies
- **Context Overlap**: Measures grounding in source material

## 🛠️ Extending the Framework

### Adding New Components

1. **New Chunker**: Inherit from `BaseChunker` in `chunking/base.py`
2. **New Embeddings**: Inherit from `BaseEmbeddings` in `embeddings/base.py`
3. **New Vector DB**: Inherit from `BaseVectorDB` in `vectordb/base.py`
4. **New LLM**: Inherit from `BaseLLM` in `llm/base.py`

### Custom Evaluation Metrics

Add new metrics to `evaluation/metrics.py` or create specialized metric classes.

## 📝 Data Formats

### QA Dataset Format
```json
[
  {
    "question": "What is the capital of France?",
    "answer": "Paris",
    "context": "France is a country in Europe. Its capital is Paris."
  }
]
```

### Document Metadata Format
```json
{
  "id": "doc_001",
  "title": "Document Title",
  "source": "path/to/document.pdf",
  "page": 1,
  "chunk_id": "chunk_001"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code follows the existing style
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🔗 References

- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [BGE Embeddings](https://github.com/FlagOpen/FlagEmbedding)
- [FAISS](https://github.com/facebookresearch/faiss)
- [ChromaDB](https://www.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)

---

## 📊 **Detailed Performance Comparison Methodology**

This section provides a comprehensive explanation of how different RAG pipeline configurations are systematically compared and evaluated.

### **🔬 Comparison Framework Overview**

Our comparison methodology uses a **controlled variable approach** where we test different pipeline configurations while keeping other variables constant. This allows us to isolate the impact of specific components (chunking strategies, embedding models, retrieval methods) on overall performance.

#### **Test Matrix Design**
```
Configuration Variables:
├── Chunking Strategy: [Fixed, Semantic, Structure-Aware, Hybrid]
├── Embedding Model: [OpenAI, BGE, Sentence-Transformers, SciBERT]  
├── Vector Database: [FAISS, ChromaDB, Milvus, Qdrant]
└── Retrieval Method: [Dense, BM25, Hybrid, Reranked]

Total Possible Combinations: 4 × 4 × 4 × 4 = 256 configurations
Selected Test Configurations: 7-10 strategic combinations
```

### **📝 Evaluation Dataset Design**

#### **Industrial Document Test Questions**
Our evaluation uses **30 carefully crafted questions** designed to test different aspects of RAG performance on technical/industrial content:

**Question Categories:**
- **Table Extraction (40%)**: Questions requiring precise data lookup from tables
  - Example: *"For Tubing Size O.D. 1.315", what is the "Gallons Per Linear Foot" for a 4 3/4 inch hole?"*
- **Comparison Analysis (23%)**: Questions requiring comparative reasoning
  - Example: *"Which configuration provides higher Cu. Ft. Per Lin. Ft. for a 6 1/2-inch hole?"*
- **Technical Specifications (17%)**: Questions about technical details and notes
  - Example: *"What does the triple asterisk note say regarding Buttress joint displacement?"*
- **Calculation/Formula (13%)**: Questions requiring mathematical understanding
  - Example: *"Calculate the difference in Gallons Per Lin. Ft. between Two and Three Strings"*
- **Procedural Knowledge (7%)**: Questions about methods and procedures
  - Example: *"What is the step-by-step method for calculating tank volume?"*

#### **Difficulty Distribution**
- **Easy (30%)**: Direct lookup questions with clear answers
- **Medium (50%)**: Questions requiring some interpretation or calculation
- **Hard (20%)**: Complex questions requiring multi-step reasoning or cross-referencing

### **📊 Comprehensive Metrics System**

Each pipeline configuration is evaluated using **four main metric categories**, each contributing to an overall composite score:

#### **1. Retrieval Quality Metrics (25% weight)**
- **Precision@5**: Percentage of top-5 retrieved chunks that are actually relevant
- **Recall@5**: Percentage of relevant chunks found in top-5 results
- **Mean Reciprocal Rank (MRR)**: Average of 1/rank for the first relevant result
- **Average Retrieval Score**: Mean similarity/relevance score of retrieved chunks

```python
# Example Calculation
retrieval_score = (
    precision_at_5 * 0.4 +
    recall_at_5 * 0.3 +
    mrr * 0.3
)
```

#### **2. Answer Quality Metrics (35% weight)**
- **Token-level F1 Score**: Precision and recall of word overlap with expected answers
- **BLEU Score**: N-gram overlap similarity measure
- **Semantic Similarity**: Cosine similarity between answer embeddings
- **Answer Completeness**: Percentage of key concepts covered
- **Context Adherence**: How well the answer stays grounded in retrieved context

```python
# Example Calculation
quality_score = (
    f1_score * 0.3 +
    semantic_similarity * 0.3 +
    answer_completeness * 0.2 +
    context_adherence * 0.2
)
```

#### **3. Factual Accuracy Metrics (30% weight)**
- **Entity Accuracy**: Percentage of named entities that appear in source context
- **Numerical Accuracy**: Percentage of numbers that match source data
- **Factual Consistency**: Overall consistency with source material
- **Hallucination Risk**: Inverse measure of factual errors and unsupported claims

```python
# Example Calculation
accuracy_score = (
    entity_accuracy * 0.3 +
    numerical_accuracy * 0.3 +
    factual_consistency * 0.2 +
    (1 - hallucination_risk) * 0.2
)
```

#### **4. Performance Metrics (10% weight)**
- **Response Time**: Average time per question (retrieval + generation)
- **Memory Usage**: Peak memory consumption during processing
- **Tokens per Second**: Processing efficiency measure
- **Setup Time**: Time required to index and prepare the system

```python
# Example Calculation (normalized)
performance_score = (
    (1 - normalized_response_time) * 0.6 +
    (1 - normalized_memory_usage) * 0.4
)
```

### **🎯 Composite Scoring and Grading**

#### **Overall Score Calculation**
```python
composite_score = (
    retrieval_score * 0.25 +      # Retrieval Quality
    quality_score * 0.35 +        # Answer Quality  
    accuracy_score * 0.30 +       # Factual Accuracy
    performance_score * 0.10      # Performance
)
```

#### **Grading Scale**
The composite score (0-1 scale) is converted to letter grades for easy interpretation:
- **A+ (0.95-1.00)**: Exceptional performance across all metrics
- **A (0.90-0.94)**: Excellent performance with minor weaknesses
- **A- (0.85-0.89)**: Very good performance, suitable for production
- **B+ (0.80-0.84)**: Good performance with some areas for improvement
- **B (0.75-0.79)**: Acceptable performance for most use cases
- **B- (0.70-0.74)**: Below average, needs optimization
- **C+ (0.65-0.69)**: Poor performance, significant issues
- **C (0.60-0.64)**: Very poor performance
- **C- (0.55-0.59)**: Unacceptable for production use
- **D (0.00-0.54)**: Failed configuration

### **⚖️ Comparative Analysis Process**

#### **Head-to-Head Comparison**
For any two configurations (A vs B), we perform direct metric-by-metric comparison:

```python
comparison_result = {
    'overall_winner': 'Config A' if score_A > score_B else 'Config B',
    'confidence': abs(score_A - score_B),
    'metric_winners': {
        'retrieval': winner_by_retrieval_score,
        'quality': winner_by_quality_score,
        'accuracy': winner_by_accuracy_score,
        'performance': winner_by_performance_score
    },
    'strength_areas': ['List of metrics where each config excels'],
    'recommendation': 'Context-specific recommendation based on use case'
}
```

#### **Statistical Significance**
- **Minimum Score Difference**: 0.05 points required for meaningful difference
- **Category Analysis**: Performance breakdown by question category
- **Consistency Check**: Standard deviation across question types
- **Outlier Detection**: Identification of questions where configuration performs unusually well/poorly

### **📈 Reporting and Visualization**

#### **Automated Report Generation**
The system generates multiple output formats:

1. **Executive Summary JSON**
   - Overall winners and scores
   - Key performance differences
   - Recommendations by use case

2. **Detailed Analysis CSV**
   - Question-by-question results
   - All metric scores
   - Performance timing data

3. **Visual Comparisons**
   - Performance heatmaps
   - Radar charts for top configurations
   - Category-wise bar charts
   - Score distribution histograms

#### **Business-Focused Insights**
- **Production Recommendation**: Best overall configuration for deployment
- **Speed-Optimized**: Fastest configuration for real-time applications  
- **Accuracy-Optimized**: Most reliable configuration for critical applications
- **Cost-Benefit Analysis**: Performance vs computational cost trade-offs

### **🔍 Quality Assurance and Validation**

#### **Methodology Validation**
- **Reproducibility**: All tests use fixed random seeds and deterministic processes
- **Cross-Validation**: Results validated across multiple test runs
- **Baseline Comparison**: Performance compared against industry-standard baselines
- **Error Analysis**: Detailed examination of failure cases and edge scenarios

#### **Bias Mitigation**
- **Question Diversity**: Balanced representation across difficulty levels and categories
- **Source Material**: Comprehensive coverage of document sections and table types
- **Evaluation Blind Spots**: Identification of potential metric limitations
- **Human Validation**: Sample of results reviewed for qualitative accuracy

### **💼 Business Value and ROI Analysis**

#### **Decision Support Framework**
The comparison methodology provides executives with:

1. **Quantitative Justification**: Numerical scores supporting technology choices
2. **Risk Assessment**: Identification of failure modes and reliability concerns  
3. **Scalability Analysis**: Performance implications for larger document sets
4. **Resource Planning**: Understanding of computational and infrastructure requirements

#### **Implementation Roadmap**
Based on comparison results, the framework recommends:
- **Phase 1**: Deploy highest-scoring configuration for pilot testing
- **Phase 2**: A/B test top 2-3 configurations with real users
- **Phase 3**: Scale winning configuration to full production
- **Ongoing**: Monitor performance and re-evaluate as new techniques emerge

### **🎯 Expected Outcomes and Success Criteria**

#### **Performance Benchmarks**
For industrial/technical document RAG systems:
- **Minimum Acceptable Score**: 0.75 (B grade)
- **Production-Ready Score**: 0.80+ (B+ or higher)
- **Excellence Threshold**: 0.90+ (A grade)

#### **Category-Specific Targets**
- **Table Extraction**: >0.85 accuracy for numerical lookups
- **Comparison Questions**: >0.80 for logical reasoning tasks
- **Technical Specifications**: >0.90 for direct factual queries
- **Response Time**: <3 seconds average per question

This comprehensive methodology ensures objective, reproducible, and business-relevant evaluation of RAG pipeline performance, enabling data-driven decisions for production deployment.

---

For questions and support, please open an issue on the repository.