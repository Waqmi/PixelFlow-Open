@echo off
setlocal
cd /d "%~dp0\..\.."

set "VENV_DIR=.venv-windows"
set "PYTHON_BIN=%VENV_DIR%\Scripts\python.exe"

if not exist "%PYTHON_BIN%" (
    echo Creating Windows x64 virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        exit /b 1
    )
)

echo Installing Windows build dependencies...
"%PYTHON_BIN%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%PYTHON_BIN%" -m pip install -r "build\windows\requirements-windows.txt"
if errorlevel 1 exit /b 1

echo Building PixelFlow-Windows-x64.exe...
"%PYTHON_BIN%" -m PyInstaller --noconfirm --clean "build\windows\PixelFlow-Windows-x64.spec"
if errorlevel 1 (
    echo Windows build failed.
    exit /b 1
)

echo.
echo Windows application generated:
echo %CD%\dist\PixelFlow-Windows-x64.exe
endlocal
