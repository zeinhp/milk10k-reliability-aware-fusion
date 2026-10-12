@echo off
REM Run this on the lab PC (local, 2x GTX 1080) after LUNA16's "dev7b_noweight" run on GPU1
REM finishes (ETA per your notes: ~15:30 WIB). Can run at the same time as run_gpu0_1080.bat in
REM a separate Command Prompt window -- they use different --device values so they won't collide.
REM
REM IMPORTANT: do NOT also set CUDA_VISIBLE_DEVICES here -- see the comment in
REM run_gpu0_1080.bat for why. Only --device selects the card in this project's scripts.

REM Setup: create the venv if missing, then ALWAYS (re)install requirements before training.
REM (Previously skipped install whenever the venv folder already existed, which left a
REM half-set-up venv missing pyyaml etc. broken forever -- pip install is a cheap no-op when
REM everything is already satisfied.)

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

echo Starting GPU1 (1080 #2) unimodal runs...
python run_all.py --group gpu1_1080 --device cuda:1 >> ..\outputs\logs\gpu1_1080_run.log 2>&1

echo Done. Check ..\outputs\logs\gpu1_1080_run.log for the full log.
pause
