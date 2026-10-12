@echo off
REM Diagnostic script -- figures out exactly why "import yaml" fails inside the venv.
REM Run this directly (double-click) on the lab PC. Everything is logged to
REM ..\outputs\logs\diagnose_venv.log so it can be inspected afterward even though the pip
REM install step is not interactive here.

cd /d "%~dp0..\src"
if not exist "..\outputs\logs" mkdir "..\outputs\logs"

set LOG=..\outputs\logs\diagnose_venv.log
echo ==== diagnose_venv run ==== > "%LOG%"

if not exist "..\venv\Scripts\python.exe" (
    echo Creating venv... >> "%LOG%"
    python -m venv ..\venv >> "%LOG%" 2>&1
)

call ..\venv\Scripts\activate.bat

echo. >> "%LOG%"
echo -- which python/pip -- >> "%LOG%"
where python >> "%LOG%" 2>&1
where pip >> "%LOG%" 2>&1
python -c "import sys; print('sys.executable =', sys.executable)" >> "%LOG%"

echo. >> "%LOG%"
echo -- pip install output -- >> "%LOG%"
pip install --upgrade pip >> "%LOG%" 2>&1
pip install -r ..\requirements.txt >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo -- pip show pyyaml -- >> "%LOG%"
pip show pyyaml >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo -- import test -- >> "%LOG%"
python -c "import yaml; print('YAML OK', yaml.__version__)" >> "%LOG%" 2>&1

echo Done. See %LOG%
type "%LOG%"
pause
