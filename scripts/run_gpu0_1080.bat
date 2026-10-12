@echo off
REM Run this on the lab PC (local, 2x GTX 1080) after LUNA16's "dev7_local" run on GPU0
REM finishes (ETA per your notes: ~15:00 WIB).
REM
REM IMPORTANT: do NOT also set CUDA_VISIBLE_DEVICES here. train.py/run_all.py select the GPU via
REM the --device flag directly (both cards stay visible to this one Python process). Setting
REM CUDA_VISIBLE_DEVICES=0 as well would remap that card to "cuda:0" from PyTorch's point of view
REM regardless of what you pass to --device, and can point this run at the wrong card.
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

echo Starting GPU0 (1080 #1) unimodal runs...
python run_all.py --group gpu0_1080 --device cuda:0 >> ..\outputs\logs\gpu0_1080_run.log 2>&1

echo Done. Check ..\outputs\logs\gpu0_1080_run.log for the full log.
pause
