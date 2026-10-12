@echo off
REM Phase 4 evaluation: runs evaluate_baselines.py against all 75 baseline checkpoints (15
REM unimodal B1 + 60 naive_concat/late_fusion/gated_fusion B2/B3/B4), each at the single
REM modality_config it was trained on. Read-only inference, resume-safe (skips rows already in
REM the output CSV), so safe to just re-run this if interrupted.
REM
REM Both 1080s are free right now (baseline_multimodal training finished on both), so this runs
REM on GPU0 alone -- 75 cells of pure inference is small compared to the 210-cell reliability_aware
REM sweep that already ran fine on one GPU, no need to split across two.

cd /d "%~dp0..\src"
call ..\venv\Scripts\activate.bat

if not exist "..\outputs\logs" mkdir "..\outputs\logs"

echo Starting baseline evaluation sweep (B1 unimodal + B2/B3/B4 baseline_multimodal) on cuda:0...
python evaluate_baselines.py --device cuda:0 >> ..\outputs\logs\evaluate_baselines_run.log 2>&1

echo Done. Check ..\outputs\logs\evaluate_baselines_run.log and ..\outputs\results\evaluate_baselines_sweep.csv
pause
