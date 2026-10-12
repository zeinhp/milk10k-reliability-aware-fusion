@echo off
REM FULL RUN: baseline_multimodal (naive_concat/late_fusion/gated_fusion), all 60 runs split
REM 30/30 across both 1080s (this is GPU1's half -- see run_all.py's
REM split_baseline_multimodal_across_two_gpus / group gpu1_1080_base). Run at the same time as
REM run_gpu0_1080_baseline.bat in a separate Command Prompt window -- different --device values
REM so they won't collide with each other. Uses configs/config_gpu1080_baseline.yaml
REM (batch_size=16).
REM
REM SKIP_N=0 -- MUST MATCH run_gpu0_1080_baseline.bat's SKIP_N EXACTLY (both scripts slice the
REM same 60-run list from the same offset, then split it between them -- mismatched SKIP_N values
REM would create overlap or gaps).
REM
REM This is safe as long as A4000 (lab.jupyterhub.my.id) stays down/unreachable: as of when this
REM was set, it had been 502 for 24+ hours and nothing is writing to any baseline_multimodal
REM checkpoint right now. THE ONE THING TO WATCH: if A4000 comes back online WHILE this is still
REM running, do NOT just restart A4000's own baseline_multimodal training -- it could pick a run
REM that's mid-training on a 1080 right now (not yet checkpointed) and corrupt that one file via
REM concurrent writes. If A4000 comes back before this finishes, check progress here first (grep
REM -c "Done. Best" outputs\logs\gpu0_1080_base_run.log + outputs\logs\gpu1_1080_base_run.log) and
REM only hand A4000 the runs neither 1080 has finished yet -- ask for a recomputed skip/allow-list
REM rather than guessing.
set SKIP_N=0

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

if "%SKIP_N%"=="__SET_ME__" (
    echo SKIP_N is not set. Read the comment at the top of this file and set it before running.
    pause
    exit /b 1
)

if not exist "..\outputs\logs" mkdir "..\outputs\logs"

echo Starting GPU1 (1080 #2) baseline_multimodal overflow runs, skipping first %SKIP_N%...
python run_all.py --group gpu1_1080_base --device cuda:1 --baseline_skip_first_n %SKIP_N% >> ..\outputs\logs\gpu1_1080_base_run.log 2>&1

echo Done. Check ..\outputs\logs\gpu1_1080_base_run.log for the full log.
pause
