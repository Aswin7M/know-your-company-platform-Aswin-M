@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" goto nosetup
call ".venv\Scripts\activate.bat"

if not exist "data\companies" mkdir "data\companies"
if not exist "data\vector_store" mkdir "data\vector_store"
if not exist ".env" if exist ".env.example" copy ".env.example" ".env" >nul

echo Checking Ollama at http://localhost:11434 ...
where curl >nul 2>nul
if errorlevel 1 goto launch
curl -s -m 3 http://localhost:11434/api/tags >nul 2>nul
if errorlevel 1 goto noollama
echo Ollama is running.
goto launch

:noollama
echo.
echo [WARNING] Ollama is not reachable at http://localhost:11434.
echo The app will still start, but AI answers will not work until Ollama is running.
echo Install it from https://ollama.com/download and run:  ollama pull qwen3:1.7b
echo If you use a different address, set OLLAMA_BASE_URL in .env and ignore this warning.
echo.

:launch
echo Starting Know Your Company. Your browser should open automatically.
streamlit run app.py
pause
exit /b 0

:nosetup
echo [ERROR] The virtual environment was not found. Run setup_windows.bat first.
pause
exit /b 1
