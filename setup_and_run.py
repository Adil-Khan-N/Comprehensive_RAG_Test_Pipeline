#!/usr/bin/env python3
"""
Simple setup and runner for RAG Framework with Gemini Pro

This script will:
1. Install required dependencies
2. Validate configuration
3. Run comprehensive RAG tests
4. Generate comparison report
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_gemini.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    return True

def check_env_file():
    """Check if .env file exists and has required values."""
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found!")
        print("   Please create .env file with your Gemini API key")
        return False
    
    # Read .env file to check if API key is set
    with open(env_path, 'r') as f:
        content = f.read()
    
    if 'GEMINI_API_KEY=your_gemini_api_key_here' in content:
        print("❌ Please update your GEMINI_API_KEY in .env file")
        return False
    
    if 'GEMINI_API_KEY=' not in content:
        print("❌ GEMINI_API_KEY not found in .env file")
        return False
        
    print("✅ .env file configured")
    return True

def check_test_document():
    """Check if test document exists."""
    doc_path = Path("data/marker_md/RedBook_Markdown.md")
    if not doc_path.exists():
        print(f"❌ Test document not found: {doc_path}")
        print("   Please copy your RedBook_Markdown.md to data/marker_md/ folder")
        return False
    print("✅ Test document found")
    return True

def run_tests(scenario="quick", max_questions=5):
    """Run the Gemini tests with specified parameters."""
    print(f"🚀 Running {scenario} test scenario with {max_questions} questions...")
    try:
        subprocess.check_call([sys.executable, "simple_test.py", "--scenario", scenario, "--max-questions", str(max_questions)])
        print("✅ Tests completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")
        return False

def main():
    """Main setup and run function."""
    print("🔥 RAG Framework - Gemini Pro Setup & Test Runner")
    print("=" * 50)
    
    # Step 1: Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Step 2: Check environment configuration
    if not check_env_file():
        print("\n📝 To fix this:")
        print("1. Edit the .env file")
        print("2. Replace 'your_gemini_api_key_here' with your actual API key")
        print("3. Get API key from: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    # Step 3: Check test document
    if not check_test_document():
        print("\n📝 To fix this:")
        print("1. Create folder: data/marker_md/")
        print("2. Copy your RedBook_Markdown.md file to that folder")
        sys.exit(1)
    
    # Step 4: Ask user for test scenario
    print("\n🎯 Choose test scenario:")
    print("1. quick        - 5 questions, single config (2-3 minutes)")
    print("2. standard     - 15 questions, basic comparison (5-10 minutes)")
    print("3. comprehensive - 30 questions, all combinations (15-30 minutes)")
    
    choice = input("\nEnter choice (1-3) or scenario name [quick]: ").strip()
    
    scenario_map = {
        '1': 'quick',
        '2': 'standard', 
        '3': 'comprehensive',
        '': 'quick'
    }
    
    scenario = scenario_map.get(choice, choice)
    if scenario not in ['quick', 'standard', 'comprehensive']:
        scenario = 'quick'
        
    # Set appropriate question counts based on scenario
    max_questions = {
        'quick': 5,
        'standard': 15,
        'comprehensive': 30
    }.get(scenario, 5)
    
    # Step 5: Run tests
    if run_tests(scenario, max_questions):
        print("\n🎉 All tests completed successfully!")
        print("📊 Check the 'results/' folder for detailed reports")
        if scenario == 'comprehensive':
            print("📈 Look for comprehensive analysis files:")
            print("   - comprehensive_results_YYYYMMDD_HHMMSS.json")
            print("   - comprehensive_report_YYYYMMDD_HHMMSS.md")
        else:
            print("📈 Look for files named with today's timestamp:")
            print("   - gemini_test_results_SCENARIO_YYYYMMDD_HHMMSS.json")
            print("   - gemini_test_report_SCENARIO_YYYYMMDD_HHMMSS.md")
    else:
        print("\n❌ Tests failed - check the log files for details")
        sys.exit(1)

if __name__ == "__main__":
    main()