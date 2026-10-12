@echo off
REM Phase 6-7 evaluation sweep (7 modality configs x ablation stages x seed) for the
REM reliability_aware model -- see src/evaluate.py for full details. Read-only inference over
REM existing checkpoints on the TEST split, does not train or modify anything, does not touch any
REM checkpoint. Safe to run any time; both 1080s are idle (all 30/30 reliability_aware overflow
REM training is done) and this doesn't touch/need A4000 at all.
REM
REM This is GPU1's HALF of the sweep: seeds 1, 3. GPU0 (run_evaluate_gpu0.bat) does seeds 0, 2, 4.
REM Writes to a SEPARATE csv (evaluate_reliability_aware_sweep_gpu1.csv) so the two GPUs never
REM append to the same file concurrently -- merge the two CSVs afterward (they're disjoint by
REM seed, so a plain concatenation, e.g. via pandas.concat, is fine).
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

echo Running Phase 6-7 evaluation sweep on GPU1 (seeds 1, 3)...
python evaluate.py --device cuda:1 --seeds 1,3 --csv_suffix _gpu1 > ..\outputs\logs\evaluate_gpu1_run.log 2>&1

echo Done. Check ..\outputs\logs\evaluate_gpu1_run.log for progress, and
echo ..\outputs\results\evaluate_reliability_aware_sweep_gpu1.csv for the results table.
pause
