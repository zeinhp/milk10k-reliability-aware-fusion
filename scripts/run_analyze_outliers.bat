@echo off
REM Read-only error analysis for the two outlier runs flagged in the 18 Aug recap: seed2/A5
REM (val macro F1 0.4427) and seed3/Full (0.4332), compared against seed0 at the same stages.
REM Loads the existing *_best.pt checkpoints and re-runs inference on each seed's val split --
REM does NOT train or modify anything, does NOT touch any checkpoint file, safe to run any time
REM on either 1080 (both are idle now, all 30/30 reliability_aware overflow runs are done) without
REM any risk of colliding with A4000 or anything else. See src/analyze_outliers.py for details.
REM Output: outputs/results/outlier_analysis.md and outlier_analysis.json.
REM
REM Setup: create the venv if missing, then ALWAYS (re)install requirements before training.
REM This used to skip installing when the venv folder already existed, which silently left a
REM half-set-up venv (missing pyyaml etc.) broken forever. pip install is a no-op for packages
REM already satisfied, so running it every time is cheap and fixes that failure mode for good.

cd /d "%~dp0..\src"

if not exist "..\venv\Scripts\python.exe" (
    echo Creating venv...
    python -m venv ..\venv
)
call ..\venv\Scripts\activate.bat
echo Installing/checking requirements...
pip install -r ..\requirements.txt

REM requirements.txt just says "torch>=2.1" with no index, so a plain `pip install` can silently
REM resolve to a CPU-only build (pip then reports it "already satisfied" forever after, even
REM though it can't see the GPUs at all). Detect that specific case (or a broken/half-installed
REM torch that fails to even import, e.g. WinError 127 loading c10_cuda.dll from a stale/mismatched
REM torch+torchvision pairing) and force a clean, version-pinned CUDA reinstall -- torch==2.3.1 /
REM torchvision==0.18.1 for cu121 is the exact pairing already confirmed working (RTX A4000), and
REM cu121 wheels still support the GTX 1080's Pascal architecture (sm_61).
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
        echo STILL BROKEN after clean reinstall -- stopping instead of wasting GPU time on CPU/failed runs.
        pause
        exit /b 1
    )
)

if not exist "..\outputs\logs" mkdir "..\outputs\logs"
if not exist "..\outputs\results" mkdir "..\outputs\results"

echo Running outlier analysis (read-only inference, no training)...
python analyze_outliers.py --device cuda:0 > ..\outputs\logs\analyze_outliers_run.log 2>&1

echo Done. Check ..\outputs\logs\analyze_outliers_run.log for console output, and
echo ..\outputs\results\outlier_analysis.md for the full per-class / confusion-matrix report.
pause
