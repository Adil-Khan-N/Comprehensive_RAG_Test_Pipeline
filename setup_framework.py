"""Setup script to prepare the RAG framework for testing."""

import os
import sys
from pathlib import Path


def create_init_files():
    """Create __init__.py files for proper module importing."""
    
    # Get the framework root directory
    framework_root = Path(__file__).parent
    
    # List of directories that need __init__.py files
    module_dirs = [
        'chunking',
        'embeddings', 
        'vectordb',
        'retriever',
        'llm',
        'evaluation'
    ]
    
    for dir_name in module_dirs:
        init_file = framework_root / dir_name / '__init__.py'
        if not init_file.exists():
            with open(init_file, 'w') as f:
                f.write(f'"""RAG Framework - {dir_name} module."""\n')
            print(f"✅ Created {init_file}")
        else:
            print(f"📁 {init_file} already exists")


def check_dependencies():
    """Check if required packages are available."""
    
    required_packages = [
        'numpy',
        'pandas', 
        'scikit-learn',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - Available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True


def verify_file_structure():
    """Verify that all required files exist."""
    
    framework_root = Path(__file__).parent
    
    required_files = [
        'data/test_questions.json',
        'evaluation/concrete_metrics.py',
        'enhanced_comparison.py'
    ]
    
    all_exist = True
    
    for file_path in required_files:
        full_path = framework_root / file_path
        if full_path.exists():
            print(f"✅ {file_path} - Found")
        else:
            print(f"❌ {file_path} - Missing")
            all_exist = False
    
    return all_exist


def create_results_directory():
    """Create results directory if it doesn't exist."""
    
    framework_root = Path(__file__).parent
    results_dir = framework_root / 'results'
    
    if not results_dir.exists():
        results_dir.mkdir()
        print(f"✅ Created results directory: {results_dir}")
    else:
        print(f"📁 Results directory already exists: {results_dir}")


def main():
    """Run setup checks and preparations."""
    
    print("🔧 RAG Framework Setup")
    print("=" * 40)
    
    print("\\n📦 Creating module __init__.py files...")
    create_init_files()
    
    print("\\n📋 Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\\n📁 Verifying file structure...")
    files_ok = verify_file_structure()
    
    print("\\n📊 Creating results directory...")
    create_results_directory()
    
    print("\\n" + "=" * 40)
    
    if deps_ok and files_ok:
        print("✅ Setup complete! Ready to run comparisons.")
        print("\\nNext steps:")
        print("1. Copy your RedBook_Markdown.md to data/marker_md/")
        print("2. Run: python enhanced_comparison.py")
    else:
        print("⚠️ Setup incomplete. Please address the issues above.")
        
    return deps_ok and files_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)