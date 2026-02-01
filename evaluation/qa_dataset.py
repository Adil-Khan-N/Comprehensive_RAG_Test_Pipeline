"""QA dataset handling for evaluation."""

from typing import List, Dict, Any, Optional
import json


class QADataset:
    """Question-Answer dataset for RAG evaluation."""
    
    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path
        self.questions = []
        self.answers = []
        self.contexts = []
        
        if dataset_path:
            self.load_dataset(dataset_path)
    
    def load_dataset(self, dataset_path: str) -> None:
        """Load QA dataset from file.
        
        Args:
            dataset_path: Path to dataset file (JSON format)
        """
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for item in data:
                self.questions.append(item.get('question', ''))
                self.answers.append(item.get('answer', ''))
                self.contexts.append(item.get('context', ''))
                
        except FileNotFoundError:
            print(f"Dataset file not found: {dataset_path}")
        except json.JSONDecodeError:
            print(f"Invalid JSON format in dataset file: {dataset_path}")
    
    def add_qa_pair(self, question: str, answer: str, context: str = "") -> None:
        """Add a question-answer pair to the dataset.
        
        Args:
            question: Question text
            answer: Answer text
            context: Context or source document
        """
        self.questions.append(question)
        self.answers.append(answer)
        self.contexts.append(context)
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get a sample from the dataset.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary containing question, answer, and context
        """
        if 0 <= index < len(self.questions):
            return {
                'question': self.questions[index],
                'answer': self.answers[index],
                'context': self.contexts[index]
            }
        raise IndexError("Sample index out of range")
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.questions)
    
    def save_dataset(self, output_path: str) -> None:
        """Save dataset to file.
        
        Args:
            output_path: Output file path
        """
        data = []
        for q, a, c in zip(self.questions, self.answers, self.contexts):
            data.append({
                'question': q,
                'answer': a,
                'context': c
            })
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)