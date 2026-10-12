@echo off
REM GPU1 (1080 #2) is stuck erroring on every run with "CUDA error: invalid device ordinal" /
REM sometimes crashing mid-training -- cuda:1 on this PC is currently unusable for training even
REM after a full reboot. GPU0 (1080 #1) already finished its own 30/30 baseline_multimodal runs
REM (see gpu0_1080_base_run.log), so it's free to pick up GPU1's remaining 30-run half instead --
REM same pattern used earlier to hand GPU1's share to the A4000 JupyterHub machine.
REM
REM This runs the EXACT SAME 30-command list as gpu1_1080_base (group "gpu1_1080_base",
REM baseline_skip_first_n=0 -- the run_all.py flag just picks which commands, not which device),
REM but forces --device cuda:0 instead of cuda:1. run_all.py's main() appends --device at the very
REM end of every command, overriding whatever device the group builder assigned internally, so this
REM is safe and does exactly what it looks like.
REM
REM No collision risk with GPU0's own already-finished 30 runs: gpu0_1080_base and gpu1_1080_base
REM are complementary halves of the same 60-run list (even vs odd indices), so there is zero overlap
REM in which (family, modality, seed) checkpoints they touch. The one run GPU1 itself already
REM finished locally before it started erroring (naive_concat C,D seed1) will just fast-skip via
REM train.py's normal resume-checkpoint logic -- no wasted retraining.
REM
REM IMPORTANT: do NOT also set CUDA_VISIBLE_DEVICES here -- see run_gpu0_1080_baseline.bat's comment
REM for why (train.py/run_all.py pick the GPU via --device directly).

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
        echo STILL BROKEN after clean reinstall -- stopping instead of wasting GPU time on CPU/failed runs.
        pause
        exit /b 1
    )
)

if not exist "..\outputs\logs" mkdir "..\outputs\logs"

echo Starting GPU0 (1080 #1) picking up GPU1's baseline_multimodal share on cuda:0...
python run_all.py --group gpu1_1080_base --device cuda:0 --baseline_skip_first_n 0 >> ..\outputs\logs\gpu0_pickup_gpu1_run.log 2>&1

echo Done. Check ..\outputs\logs\gpu0_pickup_gpu1_run.log for the full log.
pause
