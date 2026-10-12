@echo off
REM FULL RUN: baseline_multimodal (naive_concat/late_fusion/gated_fusion), all 60 runs split
REM 30/30 across both 1080s (this is GPU0's half -- see run_all.py's
REM split_baseline_multimodal_across_two_gpus / group gpu0_1080_base). Uses
REM configs/config_gpu1080_baseline.yaml (batch_size=16).
REM
REM SKIP_N=0 -- runs the WHOLE list, no skipping. This is safe as long as A4000
REM (lab.jupyterhub.my.id) stays down/unreachable: as of when this was set, it had been 502 for
REM 24+ hours and nothing is writing to any baseline_multimodal checkpoint right now, so there is
REM no concurrent-write risk. THE ONE THING TO WATCH: if A4000 comes back online WHILE this is
REM still running, do NOT just restart A4000's own baseline_multimodal training -- it could pick a
REM run that's mid-training on this 1080 right now (not yet checkpointed) and corrupt that one
REM file via concurrent writes. If A4000 comes back before this finishes, check progress here
REM first (grep -c "Done. Best" outputs\logs\gpu0_1080_base_run.log +
REM outputs\logs\gpu1_1080_base_run.log) and only hand A4000 the runs neither 1080 has finished
REM yet -- ask for a recomputed skip/allow-list rather than guessing.
set SKIP_N=0

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

if "%SKIP_N%"=="__SET_ME__" (
    echo SKIP_N is not set. Read the comment at the top of this file and set it before running.
    pause
    exit /b 1
)

if not exist "..\outputs\logs" mkdir "..\outputs\logs"

echo Starting GPU0 (1080 #1) baseline_multimodal overflow runs, skipping first %SKIP_N%...
python run_all.py --group gpu0_1080_base --device cuda:0 --baseline_skip_first_n %SKIP_N% >> ..\outputs\logs\gpu0_1080_base_run.log 2>&1

echo Done. Check ..\outputs\logs\gpu0_1080_base_run.log for the full log.
pause
