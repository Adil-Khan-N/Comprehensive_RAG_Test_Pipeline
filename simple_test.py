#!/usr/bin/env python3
"""
Simple working RAG Framework Test with Gemini Pro

This script runs a basic test using the created framework components.
"""

import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import validate_config, GEMINI_API_KEY, TEST_DOCUMENT_PATH, OUTPUT_DIR
from llm.gemini import GeminiProLLM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'rag_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_test_questions(max_questions: int = 5) -> List[Dict]:
    """Load test questions from JSON file."""
    try:
        with open('data/test_questions.json', 'r') as f:
            questions = json.load(f)
        return questions[:max_questions]
    except FileNotFoundError:
        logger.error("Test questions file not found. Creating sample questions...")
        # Create sample questions for demonstration
        sample_questions = [
            {
                "id": 1,
                "question": "What are the main technical specifications mentioned in the document?",
                "category": "specification_extraction",
                "expected_answer_type": "list"
            },
            {
                "id": 2,
                "question": "Can you extract any numerical values or measurements?",
                "category": "numerical_extraction",
                "expected_answer_type": "numbers"
            },
            {
                "id": 3,
                "question": "What are the key procedures described in the document?",
                "category": "procedural_extraction",
                "expected_answer_type": "steps"
            },
            {
                "id": 4,
                "question": "Are there any safety requirements or warnings mentioned?",
                "category": "safety_analysis",
                "expected_answer_type": "text"
            },
            {
                "id": 5,
                "question": "What is the overall purpose or objective of this document?",
                "category": "document_summary",
                "expected_answer_type": "summary"
            }
        ]
        return sample_questions[:max_questions]

def load_test_document() -> str:
    """Load the test document content."""
    try:
        with open(TEST_DOCUMENT_PATH, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Test document not found: {TEST_DOCUMENT_PATH}")
        return "Sample document content for testing RAG pipeline."

def test_comprehensive_combinations(questions: List[Dict], document: str) -> Dict:
    """Test multiple RAG component combinations comprehensively."""
    logger.info("Starting comprehensive RAG combination testing...")
    
    # Define different combinations to test
    test_combinations = [
        {"name": "Basic_Simple", "model": "gemini-1.5-flash", "temp": 0.1, "max_tokens": 1000},
        {"name": "Balanced_Standard", "model": "gemini-1.5-flash", "temp": 0.2, "max_tokens": 1500}, 
        {"name": "Creative_Extended", "model": "gemini-1.5-flash", "temp": 0.4, "max_tokens": 2000},
        {"name": "Precise_Technical", "model": "gemini-1.5-flash", "temp": 0.05, "max_tokens": 1800},
        {"name": "Detailed_Analysis", "model": "gemini-1.5-flash", "temp": 0.15, "max_tokens": 2500},
    ]
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'total_combinations': len(test_combinations),
        'total_questions_per_combo': len(questions),
        'combination_results': [],
        'summary': {
            'best_combination': None,
            'best_score': 0,
            'avg_response_time': 0,
            'total_successful': 0,
            'total_failed': 0
        }
    }
    
    total_response_time = 0
    total_tests = 0
    
    for combo_idx, combo in enumerate(test_combinations, 1):
        logger.info(f"\n=== Testing Combination {combo_idx}/{len(test_combinations)}: {combo['name']} ===")
        logger.info(f"Config: Model={combo['model']}, Temp={combo['temp']}, MaxTokens={combo['max_tokens']}")
        
        # Initialize LLM with specific configuration
        try:
            gemini_llm = GeminiProLLM(
                api_key=GEMINI_API_KEY,
                model=combo['model'], 
                temperature=combo['temp'],
                max_tokens=combo['max_tokens']
            )
        except Exception as e:
            logger.error(f"Failed to initialize {combo['name']}: {str(e)}")
            continue
            
        combo_results = {
            'combination_name': combo['name'],
            'configuration': combo,
            'questions_tested': len(questions),
            'results': [],
            'metrics': {
                'successful_queries': 0,
                'failed_queries': 0,
                'avg_response_time': 0,
                'total_response_time': 0,
                'success_rate': 0
            },
            'errors': []
        }
        
        # Test each question with this combination
        for q_idx, question in enumerate(questions, 1):
            logger.info(f"  Testing question {q_idx}/{len(questions)}: {question['question'][:60]}...")
            
            start_time = time.time()
            try:
                response = gemini_llm.generate_with_context(
                    question=question['question'],
                    context=document[:6000]  # Use more context for comprehensive test
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # Simple quality scoring based on response length and content
                quality_score = min(100, max(10, len(response) / 10))  # Basic scoring
                if any(keyword in response.lower() for keyword in ['error', 'sorry', 'cannot', "don't know"]):
                    quality_score *= 0.5
                
                result = {
                    'question_id': question.get('id', q_idx),
                    'question': question['question'],
                    'category': question.get('category', 'general'),
                    'response': response,
                    'response_time': response_time,
                    'quality_score': quality_score,
                    'status': 'success'
                }
                
                combo_results['results'].append(result)
                combo_results['metrics']['successful_queries'] += 1
                combo_results['metrics']['total_response_time'] += response_time
                total_response_time += response_time
                total_tests += 1
                
                logger.info(f"    ✅ Completed in {response_time:.2f}s (Quality: {quality_score:.1f})")
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"    ❌ Failed: {error_msg}")
                
                result = {
                    'question_id': question.get('id', q_idx),
                    'question': question['question'],
                    'category': question.get('category', 'general'),
                    'error': error_msg,
                    'response_time': 0,
                    'quality_score': 0,
                    'status': 'failed'
                }
                
                combo_results['results'].append(result)
                combo_results['metrics']['failed_queries'] += 1
                combo_results['errors'].append(f"Q{q_idx}: {error_msg}")
            
            # Small delay between questions
            time.sleep(0.5)
        
        # Calculate combination metrics
        if combo_results['metrics']['successful_queries'] > 0:
            combo_results['metrics']['avg_response_time'] = (
                combo_results['metrics']['total_response_time'] / 
                combo_results['metrics']['successful_queries']
            )
            combo_results['metrics']['success_rate'] = (
                combo_results['metrics']['successful_queries'] / len(questions) * 100
            )
            
            # Calculate overall quality score for this combination
            successful_results = [r for r in combo_results['results'] if r['status'] == 'success']
            if successful_results:
                avg_quality = sum(r['quality_score'] for r in successful_results) / len(successful_results)
                combo_score = avg_quality * (combo_results['metrics']['success_rate'] / 100)
                
                # Track best combination
                if combo_score > all_results['summary']['best_score']:
                    all_results['summary']['best_combination'] = combo['name']
                    all_results['summary']['best_score'] = combo_score
        
        all_results['combination_results'].append(combo_results)
        all_results['summary']['total_successful'] += combo_results['metrics']['successful_queries']
        all_results['summary']['total_failed'] += combo_results['metrics']['failed_queries']
        
        logger.info(f"  ✅ {combo['name']} completed: {combo_results['metrics']['successful_queries']}/{len(questions)} success")
        
        # Longer delay between combinations to respect rate limits
        time.sleep(2)
    
    # Calculate overall metrics
    if total_tests > 0:
        all_results['summary']['avg_response_time'] = total_response_time / total_tests
    
    logger.info(f"\n🎯 Comprehensive testing completed!")
    logger.info(f"   Best combination: {all_results['summary']['best_combination']}")
    logger.info(f"   Best score: {all_results['summary']['best_score']:.2f}")
    
    return all_results
    """Test the RAG pipeline with Gemini Pro."""
    logger.info("Initializing Gemini Pro LLM...")
    
    # Initialize Gemini LLM
    gemini_llm = GeminiProLLM(
        api_key=GEMINI_API_KEY,
        temperature=0.2,
        max_tokens=1500
    )
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_questions': len(questions),
        'results': [],
        'summary': {
            'successful_queries': 0,
            'failed_queries': 0,
            'total_response_time': 0.0,
            'average_response_time': 0.0
        },
        'errors': []
    }
    
    for i, question in enumerate(questions, 1):
        logger.info(f"Processing question {i}/{len(questions)}: {question['question'][:50]}...")
        
        start_time = time.time()
        try:
            # Simple RAG simulation: use document as context and ask question
            response = gemini_llm.generate_with_context(
                question=question['question'],
                context=document[:5000]  # Use first 5000 chars as context
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            result = {
                'question_id': question['id'],
                'question': question['question'],
                'category': question['category'],
                'expected_type': question['expected_answer_type'],
                'response': response,
                'response_time': response_time,
                'status': 'success'
            }
            
            results['results'].append(result)
            results['summary']['successful_queries'] += 1
            results['summary']['total_response_time'] += response_time
            
            logger.info(f"✅ Question {i} completed in {response_time:.2f}s")
            
            # Add small delay to respect rate limits
            time.sleep(1)
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Question {i} failed: {error_msg}")
            
            result = {
                'question_id': question['id'],
                'question': question['question'],
                'category': question['category'],
                'error': error_msg,
                'status': 'failed'
            }
            
            results['results'].append(result)
            results['summary']['failed_queries'] += 1
            results['errors'].append(f"Question {i}: {error_msg}")
    
    # Calculate average response time
    if results['summary']['successful_queries'] > 0:
        results['summary']['average_response_time'] = (
            results['summary']['total_response_time'] / results['summary']['successful_queries']
        )
    
    return results

def save_comprehensive_results(results: Dict, scenario: str = 'comprehensive') -> None:
    """Save comprehensive test results with detailed analysis."""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed JSON results
    json_file = output_dir / f'comprehensive_results_{timestamp}.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # Create comprehensive comparison report
    report_content = f"""# Comprehensive RAG Framework Test Results

**Test Run**: {results['timestamp']}
**Total Combinations Tested**: {results['total_combinations']}
**Questions per Combination**: {results['total_questions_per_combo']}
**Total Tests Executed**: {results['total_combinations'] * results['total_questions_per_combo']}

## Executive Summary

🏆 **Best Performing Combination**: {results['summary']['best_combination']}
📊 **Best Score**: {results['summary']['best_score']:.2f}
⏱️ **Average Response Time**: {results['summary']['avg_response_time']:.2f}s
✅ **Total Successful**: {results['summary']['total_successful']}
❌ **Total Failed**: {results['summary']['total_failed']}

## Detailed Combination Results

"""
    
    # Sort combinations by performance
    sorted_combinations = sorted(
        results['combination_results'],
        key=lambda x: x['metrics']['success_rate'] * (sum(r.get('quality_score', 0) for r in x['results']) / len(x['results']) if x['results'] else 0),
        reverse=True
    )
    
    for i, combo in enumerate(sorted_combinations, 1):
        metrics = combo['metrics']
        config = combo['configuration']
        
        # Calculate average quality score
        successful_results = [r for r in combo['results'] if r['status'] == 'success']
        avg_quality = sum(r.get('quality_score', 0) for r in successful_results) / len(successful_results) if successful_results else 0
        
        report_content += f"""### {i}. {combo['combination_name']}

**Configuration:**
- Model: {config['model']}
- Temperature: {config['temp']}
- Max Tokens: {config['max_tokens']}

**Performance:**
- Success Rate: {metrics['success_rate']:.1f}% ({metrics['successful_queries']}/{combo['questions_tested']})
- Average Response Time: {metrics['avg_response_time']:.2f}s
- Average Quality Score: {avg_quality:.1f}/100
- Failed Queries: {metrics['failed_queries']}

**Sample Responses:**
"""
        
        # Show top 2 successful responses
        successful_responses = [r for r in combo['results'] if r['status'] == 'success'][:2]
        for resp in successful_responses:
            report_content += f"""- **Q**: {resp['question'][:80]}...
- **A**: {resp['response'][:150]}...
- **Time**: {resp['response_time']:.2f}s, **Quality**: {resp.get('quality_score', 0):.1f}

"""
        
        if combo['errors']:
            report_content += f"**Errors**: {len(combo['errors'])} failures\n"
        
        report_content += "---\n\n"
    
    # Add recommendations
    report_content += f"""## Recommendations

### 🥇 Best Overall: {results['summary']['best_combination']}
Recommended for production use based on balance of accuracy, speed, and reliability.

### ⚡ Fastest Configuration
{min(sorted_combinations, key=lambda x: x['metrics']['avg_response_time'] if x['metrics']['avg_response_time'] > 0 else float('inf'))['combination_name']} - Best for real-time applications.

### 🎯 Most Accurate Configuration  
{max(sorted_combinations, key=lambda x: x['metrics']['success_rate'])['combination_name']} - Best for critical accuracy requirements.

### 📊 Performance Matrix

| Combination | Success Rate | Avg Response Time | Avg Quality | Use Case |
|-------------|--------------|-------------------|-------------|----------|
"""
    
    for combo in sorted_combinations:
        metrics = combo['metrics']
        successful_results = [r for r in combo['results'] if r['status'] == 'success']
        avg_quality = sum(r.get('quality_score', 0) for r in successful_results) / len(successful_results) if successful_results else 0
        
        use_case = "General Purpose"
        if metrics['success_rate'] > 95:
            use_case = "High Accuracy"
        elif metrics['avg_response_time'] < 1.0:
            use_case = "Real-time"
        elif combo['configuration']['max_tokens'] > 2000:
            use_case = "Detailed Analysis"
        
        report_content += f"| {combo['combination_name']} | {metrics['success_rate']:.1f}% | {metrics['avg_response_time']:.2f}s | {avg_quality:.1f} | {use_case} |\n"
    
    report_content += f"""\n## Files Generated

- `comprehensive_results_{timestamp}.json` - Complete detailed results
- `comprehensive_report_{timestamp}.md` - This analysis report

*Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*
"""
    
    # Save comprehensive report
    report_file = output_dir / f'comprehensive_report_{timestamp}.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"Comprehensive results saved:")
    logger.info(f"  📄 Detailed results: {json_file}")
    logger.info(f"  📊 Analysis report: {report_file}")

def save_results(results: Dict, scenario: str = 'quick') -> None:
    """Save test results to files."""
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed JSON results
    json_file = output_dir / f'gemini_test_results_{scenario}_{timestamp}.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # Create summary report
    report_content = f"""# Gemini Pro RAG Test Results

**Test Run**: {results['timestamp']}
**Scenario**: {scenario}
**Total Questions**: {results['total_questions']}

## Summary

- ✅ **Successful Queries**: {results['summary']['successful_queries']}
- ❌ **Failed Queries**: {results['summary']['failed_queries']}
- ⏱️ **Average Response Time**: {results['summary']['average_response_time']:.2f} seconds
- 📊 **Success Rate**: {(results['summary']['successful_queries'] / results['total_questions'] * 100):.1f}%

## Results

"""
    
    for result in results['results']:
        if result['status'] == 'success':
            report_content += f"""### Question {result['question_id']}: {result['category']}

**Question**: {result['question']}

**Response**: {result['response'][:200]}...

**Response Time**: {result['response_time']:.2f}s

---

"""
        else:
            report_content += f"""### Question {result['question_id']}: {result['category']} ❌

**Question**: {result['question']}

**Error**: {result['error']}

---

"""
    
    if results['errors']:
        report_content += "\\n## Errors\\n\\n"
        for error in results['errors']:
            report_content += f"- {error}\\n"
    
    # Save report
    report_file = output_dir / f'gemini_test_report_{scenario}_{timestamp}.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"Results saved:")
    logger.info(f"  📄 Detailed results: {json_file}")
    logger.info(f"  📊 Summary report: {report_file}")

def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test RAG framework with Gemini Pro')
    parser.add_argument('--scenario', choices=['quick', 'standard', 'comprehensive'], default='quick')
    parser.add_argument('--max-questions', type=int, default=5, help='Maximum questions to test')
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting Gemini Pro RAG Test - Scenario: {args.scenario}")
    
    try:
        # Validate configuration
        validate_config()
        logger.info("✅ Configuration validated")
        
        # Load test data
        questions = load_test_questions(args.max_questions)
        logger.info(f"✅ Loaded {len(questions)} test questions")
        
        document = load_test_document()
        logger.info(f"✅ Loaded test document ({len(document)} characters)")
        
        # Run tests based on scenario
        if args.scenario == 'comprehensive':
            results = test_comprehensive_combinations(questions, document)
        else:
            results = test_gemini_rag_pipeline(questions, document)
        
        # Save results
        if args.scenario == 'comprehensive':
            save_comprehensive_results(results, args.scenario)
        else:
            save_results(results, args.scenario)
        
        # Print summary
        summary = results['summary']
        logger.info(f"\\n🎉 Test completed!")
        logger.info(f"   Success rate: {summary['successful_queries']}/{results['total_questions']} ({(summary['successful_queries']/results['total_questions']*100):.1f}%)")
        logger.info(f"   Average response time: {summary['average_response_time']:.2f}s")
        
        if summary['failed_queries'] > 0:
            logger.warning(f"   ⚠️  {summary['failed_queries']} queries failed")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()