"""Experiment runner for RAG framework."""

import yaml
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

# Add the parent directory to the path to import from rag_framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunking import FixedSizeChunker, SemanticChunker, HybridChunker
from embeddings import OpenAIEmbeddings, SentenceTransformersEmbeddings
from vectordb import FAISSVectorDB, ChromaVectorDB
from retriever import DenseRetriever, BM25Retriever, HybridRetriever
from llm import OpenAILLM, LocalLLM
from evaluation import QADataset, RAGMetrics


class ExperimentRunner:
    """Run RAG experiments based on configuration files."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.results = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Load experiment configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_chunker(self) -> Any:
        """Setup chunker based on configuration."""
        chunker_config = self.config['chunking']
        chunker_type = chunker_config['type']
        
        if chunker_type == 'fixed':
            return FixedSizeChunker(
                chunk_size=chunker_config.get('chunk_size', 512),
                overlap=chunker_config.get('overlap', 50)
            )
        elif chunker_type == 'semantic':
            return SemanticChunker(
                similarity_threshold=chunker_config.get('similarity_threshold', 0.8)
            )
        elif chunker_type == 'hybrid':
            return HybridChunker()
        else:
            raise ValueError(f"Unknown chunker type: {chunker_type}")
    
    def setup_embeddings(self) -> Any:
        """Setup embeddings model based on configuration."""
        embedding_config = self.config['embeddings']
        embedding_type = embedding_config['type']
        
        if embedding_type == 'openai':
            return OpenAIEmbeddings(
                model=embedding_config.get('model', 'text-embedding-ada-002')
            )
        elif embedding_type == 'sentence_transformers':
            return SentenceTransformersEmbeddings(
                model_name=embedding_config.get('model_name', 'all-MiniLM-L6-v2')
            )
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    def setup_vectordb(self, dimension: int) -> Any:
        """Setup vector database based on configuration."""
        vectordb_config = self.config['vectordb']
        vectordb_type = vectordb_config['type']
        
        if vectordb_type == 'faiss':
            return FAISSVectorDB(dimension=dimension)
        elif vectordb_type == 'chroma':
            return ChromaVectorDB(
                collection_name=vectordb_config.get('collection_name', 'default')
            )
        else:
            raise ValueError(f"Unknown vector database type: {vectordb_type}")
    
    def setup_retriever(self, embeddings, vectordb) -> Any:
        """Setup retriever based on configuration."""
        retriever_config = self.config['retriever']
        retriever_type = retriever_config['type']
        
        if retriever_type == 'dense':
            return DenseRetriever(embeddings, vectordb)
        elif retriever_type == 'bm25':
            return BM25Retriever()
        elif retriever_type == 'hybrid':
            dense_retriever = DenseRetriever(embeddings, vectordb)
            bm25_retriever = BM25Retriever()
            return HybridRetriever(dense_retriever, bm25_retriever)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
    
    def setup_llm(self) -> Any:
        """Setup language model based on configuration."""
        llm_config = self.config['llm']
        llm_type = llm_config['type']
        
        if llm_type == 'openai':
            return OpenAILLM(
                model=llm_config.get('model', 'gpt-3.5-turbo'),
                temperature=llm_config.get('temperature', 0.7)
            )
        elif llm_type == 'local':
            return LocalLLM(
                model_path=llm_config['model_path'],
                temperature=llm_config.get('temperature', 0.7)
            )
        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment."""
        print(f"Starting experiment: {self.config['experiment']['name']}")
        
        # Setup components
        chunker = self.setup_chunker()
        embeddings = self.setup_embeddings()
        vectordb = self.setup_vectordb(768)  # Default dimension
        retriever = self.setup_retriever(embeddings, vectordb)
        llm = self.setup_llm()
        
        # Load evaluation dataset
        dataset_path = self.config['evaluation']['dataset_path']
        qa_dataset = QADataset(dataset_path)
        
        # Initialize metrics
        metrics = RAGMetrics()
        
        # Run evaluation
        predictions = []
        ground_truths = []
        
        print(f"Evaluating on {len(qa_dataset)} samples...")
        
        for i in range(len(qa_dataset)):
            sample = qa_dataset.get_sample(i)
            question = sample['question']
            answer = sample['answer']
            
            # Retrieve relevant documents
            retrieved_docs = retriever.retrieve(question, k=5)
            
            # Create context from retrieved documents
            context = "\\n".join([doc.get('text', '') for doc in retrieved_docs])
            
            # Generate answer using LLM
            prompt = f"Context: {context}\\n\\nQuestion: {question}\\n\\nAnswer:"
            prediction = llm.generate(prompt)
            
            predictions.append(prediction)
            ground_truths.append(answer)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(qa_dataset)} samples")
        
        # Calculate metrics
        results = metrics.evaluate_batch(predictions, ground_truths)
        
        # Add experiment metadata
        results['experiment'] = {
            'name': self.config['experiment']['name'],
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'num_samples': len(qa_dataset)
        }
        
        self.results = results
        return results
    
    def save_results(self, output_path: str) -> None:
        """Save experiment results to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {output_path}")


def main():
    """Main function to run experiment from command line."""
    if len(sys.argv) != 2:
        print("Usage: python run_experiment.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}")
        sys.exit(1)
    
    # Run experiment
    runner = ExperimentRunner(config_file)
    results = runner.run_experiment()
    
    # Save results
    experiment_name = results['experiment']['name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/{experiment_name}_{timestamp}.json"
    runner.save_results(output_path)
    
    # Print summary
    print("\\n" + "="*50)
    print("EXPERIMENT RESULTS")
    print("="*50)
    print(f"Experiment: {experiment_name}")
    print(f"Samples: {results['experiment']['num_samples']}")
    print(f"Exact Match: {results['exact_match']:.3f}")
    print(f"F1 Score: {results['f1_score']:.3f}")
    print(f"BLEU Score: {results['bleu_score']:.3f}")
    print(f"ROUGE-L: {results['rouge_l']:.3f}")


if __name__ == "__main__":
    main()