@echo off
echo ========================================
echo RAG Pipeline Testing - All Combinations
echo ========================================
echo.
echo This will test ALL 144 possible combinations:
echo - 4 Chunking strategies
echo - 6 Embedding models  
echo - 6 Vector databases
echo.
echo Total tests: 1,440 individual tests
echo Estimated time: 45-90 minutes
echo.
echo WARNING: This is a comprehensive test that will take significant time!
echo.
pause

python test_all_combinations.py

echo.
echo ========================================
echo Testing Complete! 
echo Check the 'output' folder for results.
echo ========================================
pause