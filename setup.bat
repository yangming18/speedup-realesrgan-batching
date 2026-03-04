@echo off
REM Setup script for Video Editor application (Windows)

echo ================================================
echo Video Editor - Setup Script
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.11 from python.org
    exit /b 1
)

echo Python found
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv .venv

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies...
echo This may take several minutes...
pip install -r requirements.txt

REM Fix dependency conflicts automatically
echo.
echo Checking and fixing dependency conflicts...
python -c "from utils.dependency_patch import fix_dependencies; fix_dependencies(verbose=True)" 2>nul || echo Warning: Could not run dependency patch (will run at startup)

echo.
echo ================================================
echo Complete Setup Finished!
echo ================================================
echo.
echo Note: OpenCV compatibility patch
echo    The application automatically applies an OpenCV patch at startup
echo    to ensure compatibility between packages. No manual action needed.
echo    For details, see: docs/OPENCV_PATCH.md
echo.
echo Next steps:
echo 1. Run the application:
echo    .venv\Scripts\activate.bat
echo    python main.py
echo.
echo The app will open in your browser at http://localhost:7860
echo.

pause
