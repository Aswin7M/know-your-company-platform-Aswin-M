@echo off
setlocal
cd /d "%~dp0"

echo ==================================================
echo   Know Your Company - Windows setup
echo ==================================================
echo.

where python >nul 2>nul
if errorlevel 1 goto nopython

python --version
python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)"
if errorlevel 1 echo [WARNING] Python 3.10 to 3.13 is recommended. Other versions may fail to install dependencies.

if exist ".venv\Scripts\activate.bat" goto activate
echo Creating virtual environment in .venv ...
python -m venv .venv
if errorlevel 1 goto failed

:activate
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto failed

echo.
echo Installing dependencies. The first install downloads PyTorch and can take several minutes...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto failed

echo.
echo Creating data folders ...
if not exist "data\companies" mkdir "data\companies"
if not exist "data\vector_store" mkdir "data\vector_store"
if not exist ".env" copy ".env.example" ".env" >nul

echo.
echo ==================================================
echo   Setup complete.
echo ==================================================
echo Next steps:
echo   1. Install Ollama from https://ollama.com/download  (free, no account needed)
echo   2. Open a terminal and run:  ollama pull qwen3:1.7b
echo   3. Double-click run.bat to start the app
echo.
echo See OLLAMA_SETUP.md for details. No API keys are required.
echo No large AI models were downloaded by this script.
pause
exit /b 0

:nopython
echo [ERROR] Python was not found on your PATH.
echo Install Python 3.10 to 3.12 from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during installation, then run this script again.
pause
exit /b 1

:failed
echo.
echo [ERROR] Setup failed. Read the messages above, fix the problem and run setup_windows.bat again.
pause
exit /b 1
