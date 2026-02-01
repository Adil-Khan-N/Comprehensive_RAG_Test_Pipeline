@echo off
REM RAG Framework - One-Command Installation Script
REM ================================================
echo ================================================================================
echo ⚡ RAG FRAMEWORK - COMPREHENSIVE INSTALLATION
echo ================================================================================
echo.
echo This script will install ALL packages needed for the complete RAG framework
echo with all 144 pipeline combinations (4 chunking × 6 embeddings × 6 vector databases)
echo.
echo 📦 Installation will include:
echo    • Core dependencies (langchain, pandas, numpy, etc.)
echo    • All 6 embedding models support
echo    • All 6 vector databases
echo    • Gemini API integration
echo    • Development tools
echo.

set /p choice="Continue with installation? (y/N): "
if /i not "%choice%"=="y" (
    echo Installation cancelled.
    exit /b 0
)

echo.
echo 🚀 Starting comprehensive package installation...
echo.

REM Install from requirements.txt
echo ⬇️  Installing core packages...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo ❌ Error installing packages from requirements.txt
    echo Try manual installation: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ✅ Installation completed successfully!
echo.
echo 📋 NEXT STEPS:
echo.
echo 1. Set up your API keys in .env file:
echo    GEMINI_API_KEY=your_gemini_api_key_here
echo    OPENAI_API_KEY=your_openai_api_key_here (optional)
echo.
echo 2. Test your installation:
echo    python show_components.py        (View components)
echo    python test_quick_sample.py      (Quick test - 12 combinations)  
echo    python test_all_combinations.py  (Full test - 144 combinations)
echo.
echo 🎉 RAG Framework is ready to use!
echo.
pause