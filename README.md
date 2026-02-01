# RAG Framework - Comprehensive Testing System

A complete RAG (Retrieval-Augmented Generation) framework using LangChain with comprehensive testing of all component combinations.

## 🏗️ Framework Architecture

### Components
- **4 Chunking Strategies**: Fixed, Semantic, Structure-Aware, Hybrid
- **6 Embedding Models**: SentenceTransformers, BGE, E5, SciBERT, LayoutLM, OpenAI
- **6 Vector Databases**: FAISS, ChromaDB, Milvus, Qdrant, Weaviate, Pinecone

### Total Combinations
**144 pipeline combinations** (4 × 6 × 6) tested against 10 technical questions each = **1,440 individual tests**

## 🚀 Quick Start

### Option 1: Quick Test (Recommended First)
```bash
python test_quick_sample.py
```
- **12 representative combinations**
- **Runtime: 8-15 minutes**
- **120 individual tests**
- Great for initial validation

### Option 2: Full Comprehensive Test  
```bash
python test_all_combinations.py
```
- **All 144 combinations**
- **Runtime: 45-90 minutes**
- **1,440 individual tests**
- Complete performance matrix

### Option 3: Batch Execution (Windows)
```batch
run_all_tests.bat
```
- Same as Option 2 with automation
- Handles prerequisites and cleanup

### Option 4: View Components
```bash
python show_components.py
```
- Shows all available components
- Testing options summary
- No actual testing

## 📊 Testing Methodology

### Questions Dataset
10 technical questions covering:
- Software engineering principles
- Data structures and algorithms
- System design concepts
- Machine learning fundamentals
- Database management

### Scoring System (Advanced 4-Component)
1. **Keyword Matching** (25%): Technical term coverage
2. **Numerical Accuracy** (25%): Quantitative correctness 
3. **Semantic Relevance** (25%): Contextual understanding
4. **Completeness** (25%): Comprehensive coverage

### Output Reports
- **Detailed Results**: `output/comprehensive_test_results_YYYYMMDD_HHMMSS.csv`
- **Component Analysis**: `output/component_analysis_YYYYMMDD_HHMMSS.csv`
- **Performance Ranking**: Console display with top performers

## 📁 Project Structure

```
rag_framework/
├── chunking/              # LangChain text splitters
│   ├── fixed.py          # RecursiveCharacterTextSplitter
│   ├── semantic.py       # SemanticChunker  
│   ├── structure_aware.py # MarkdownHeaderTextSplitter
│   └── hybrid.py         # Combined approach
├── embeddings/           # LangChain embeddings
│   ├── sentence_transformers.py  # HuggingFaceEmbeddings
│   ├── bge.py           # BGE model
│   ├── e5.py            # E5 model
│   ├── scibert.py       # SciBERT model
│   ├── layoutlm.py      # LayoutLM model  
│   └── openai.py        # OpenAIEmbeddings
├── vectordb/             # LangChain vector stores
│   ├── faiss.py         # FAISS
│   ├── chroma.py        # ChromaDB
│   ├── milvus.py        # Milvus
│   ├── qdrant.py        # Qdrant
│   ├── weaviate.py      # Weaviate
│   └── pinecone.py      # Pinecone
├── comprehensive_rag_test.py      # Main testing framework
├── test_all_combinations.py       # Full test execution
├── test_quick_sample.py          # Quick test execution  
├── show_components.py            # Component inventory
├── run_all_tests.bat            # Windows batch script
└── output/                      # Test results directory
```

## 🔧 Installation

### Prerequisites
```bash
pip install langchain>=0.1.0 langchain-community langchain-text-splitters
pip install sentence-transformers transformers torch
pip install faiss-cpu chromadb
pip install openai  # If using OpenAI embeddings
```

### Optional Vector Database Dependencies
```bash
# For Milvus
pip install pymilvus

# For Qdrant  
pip install qdrant-client

# For Weaviate
pip install weaviate-client

# For Pinecone
pip install pinecone-client
```

## 📈 Example Output

```
🏆 TOP 10 PIPELINE COMBINATIONS:
═══════════════════════════════════════════════════════════════════════

Rank | Avg Score | Pipeline Configuration
-----|-----------|------------------------------------------------
  1  |   0.875   | SemanticChunker + BGEEmbeddings + ChromaVectorDB
  2  |   0.842   | HybridChunker + E5Embeddings + FAISSVectorDB  
  3  |   0.831   | SemanticChunker + E5Embeddings + QdrantVectorDB
  4  |   0.819   | StructureAwareChunker + BGEEmbeddings + MilvusVectorDB
  5  |   0.806   | HybridChunker + SentenceTransformersEmbeddings + ChromaVectorDB

📊 Component Performance Summary:
═══════════════════════════════════════════════════════════════════════

🔧 Best Chunking Strategy: SemanticChunker (Avg: 0.823)
🧠 Best Embedding Model: BGEEmbeddings (Avg: 0.815)  
🗃️ Best Vector Database: ChromaVectorDB (Avg: 0.808)

Total Combinations Tested: 144
Total Individual Tests: 1,440
Overall Success Rate: 89.2%
```

## 🎯 Use Cases

- **Performance Comparison**: Identify optimal component combinations
- **Component Analysis**: Understand individual component strengths
- **Baseline Establishment**: Set performance benchmarks
- **Configuration Selection**: Choose best setup for your use case
- **Research & Development**: Systematic evaluation of RAG architectures

## ⚠️ Important Notes

1. **Runtime**: Full testing takes 45-90 minutes. Start with quick test.
2. **Resources**: Requires sufficient RAM for vector operations
3. **Dependencies**: Some vector databases need additional setup
4. **API Keys**: OpenAI embeddings require OPENAI_API_KEY environment variable
5. **Output Size**: Full test generates ~50MB of detailed results

## 🤝 Contributing

## 🤝 Contributing

1. Add new components to respective directories
2. Update component lists in test files
3. Follow LangChain integration patterns
4. Ensure consistent scoring methodology

## 📄 License

MIT License - Feel free to use and modify for your research and projects.



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