@echo off
REM Phase 1 (dataset audit) + Phase 2 (freeze the split) -- run this ONCE before any training.
REM Uses the same venv as the training launchers (already has pandas/numpy/sklearn/pyyaml).

cd /d "%~dp0..\src"
call ..\venv\Scripts\activate.bat

if not exist "..\outputs\results" mkdir "..\outputs\results"
if not exist "..\splits" mkdir "..\splits"

echo ==== Phase 1: audit_dataset.py ====
python audit_dataset.py
if errorlevel 1 (
    echo Audit FAILED -- fix the error above before continuing. Not running make_splits.py.
    pause
    exit /b 1
)

echo.
echo ==== Phase 2: make_splits.py ====
python make_splits.py
if errorlevel 1 (
    echo make_splits.py FAILED -- see error above.
    pause
    exit /b 1
)

echo.
echo Done. Check ..\outputs\results\dataset_audit.json and ..\splits\seed*_*.csv
echo IMPORTANT: copy the whole "splits" folder as-is to every other machine (A4000 JupyterHub) --
echo do not regenerate it there, or the 3 GPUs will train/evaluate on different splits.
pause
