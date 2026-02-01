#!/usr/bin/env python3
"""
Simplified RAG Framework Test Runner with Gemini Pro

This script runs tests using the existing enhanced_comparison.py infrastructure.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from config import validate_config, TEST_SCENARIOS, OUTPUT_DIR
from evaluation.enhanced_comparison import run_comprehensive_comparison

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

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Run RAG framework tests with Gemini Pro')
    parser.add_argument(
        '--scenario', 
        choices=['quick', 'standard', 'comprehensive'],
        default='standard',
        help='Test scenario to run (default: standard)'
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting RAG Framework Test - Scenario: {args.scenario}")
    
    try:
        # Validate configuration
        validate_config()
        logger.info("✓ Configuration validated")
        
        # Get scenario config
        scenario_config = TEST_SCENARIOS[args.scenario]
        logger.info(f"✓ Running {args.scenario} scenario with {scenario_config['max_questions']} questions")
        
        # Run the comprehensive comparison using existing infrastructure
        results = run_comprehensive_comparison(
            max_questions=scenario_config['max_questions'],
            scenario_name=args.scenario
        )
        
        if results:
            logger.info(f"\n✓ Tests completed successfully!")
            logger.info(f"✓ Results saved to {OUTPUT_DIR}/")
            
            # Show quick summary if available
            if isinstance(results, dict) and 'summary' in results:
                summary = results['summary']
                logger.info(f"\n📊 Quick Summary:")
                logger.info(f"   Questions tested: {summary.get('total_questions', 'N/A')}")
                logger.info(f"   Best combination: {summary.get('best_combination', 'N/A')}")
                logger.info(f"   Best score: {summary.get('best_score', 'N/A')}")
        else:
            logger.warning("Tests completed but no results returned")
        
    except Exception as e:
        logger.error(f"Test runner failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()