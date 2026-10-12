@echo off
REM Phase 6 + Phase 8 prep, GPU1 half (seeds 1, 3). Runs on the idle 1080 while A4000 is down --
REM does not touch A4000 or any of its checkpoints, read-only inference only. Two steps in order
REM (both use this GPU, so they run sequentially, not concurrently):
REM   1. save_predictions.py -- dumps per-lesion predictions for the Full-stage model, all 7
REM      modality configs. Needed as input for bootstrap_ci.py and subgroup_analysis.py (run
REM      those separately afterward, once BOTH gpu0 and gpu1 halves of this script are done).
REM   2. stress_test.py -- controlled contradictory-evidence stress test (CLAUDE.md Phase 6):
REM      swaps one modality's evidence across lesions (same-class and cross-class), all 3 swap
REM      targets (C, D, M), Full-stage model.
REM
REM Writes to _gpu1-suffixed CSVs so this and run_gpu0_phase8.bat never touch the same file.
REM Merge afterward (disjoint by seed).

cd /d "%~dp0..\src"

if not exist "..\venv\Scripts\python.exe" (
    echo Creating venv...
    python -m venv ..\venv
)
call ..\venv\Scripts\activate.bat
echo Installing/checking requirements...
pip install -r ..\requirements.txt

python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 (
    echo Installed torch is missing/broken/CPU-only -- doing a clean CUDA reinstall...
    pip uninstall -y torch torchvision -q
    for %%D in (torch torchvision torch-*.dist-info torchvision-*.dist-info) do (
        if exist "..\venv\Lib\site-packages\%%D" rmdir /s /q "..\venv\Lib\site-packages\%%D"
    )
    pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
    python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
    if errorlevel 1 (
        echo STILL BROKEN after clean reinstall -- stopping instead of wasting GPU time.
        pause
        exit /b 1
    )
)

if not exist "..\outputs\logs" mkdir "..\outputs\logs"
if not exist "..\outputs\results" mkdir "..\outputs\results"

echo [1/2] Saving per-lesion predictions (GPU1, seeds 1,3)...
python save_predictions.py --device cuda:1 --seeds 1,3 --csv_suffix _gpu1 > ..\outputs\logs\save_predictions_gpu1_run.log 2>&1

echo [2/2] Running contradiction stress test (GPU1, seeds 1,3)...
python stress_test.py --device cuda:1 --seeds 1,3 --csv_suffix _gpu1 > ..\outputs\logs\stress_test_gpu1_run.log 2>&1

echo Done. Check ..\outputs\logs\save_predictions_gpu1_run.log and stress_test_gpu1_run.log.
pause
