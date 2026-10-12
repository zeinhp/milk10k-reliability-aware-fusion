@echo off
REM Dumps per-lesion predictions for all 75 baseline checkpoints (needed for the paired
REM bootstrap significance test against the reliability-aware model's predictions, which are
REM already saved in predictions_full_stage_gpu0.csv / _gpu1.csv). Same cost as
REM run_evaluate_baselines.bat (~15 min), just writing per-sample rows instead of aggregate
REM metrics. Resume-safe if interrupted.

cd /d "%~dp0..\src"
call ..\venv\Scripts\activate.bat

if not exist "..\outputs\logs" mkdir "..\outputs\logs"

echo Starting per-lesion baseline prediction dump on cuda:0...
python save_predictions_baselines.py --device cuda:0 >> ..\outputs\logs\save_predictions_baselines_run.log 2>&1

echo Done. Check ..\outputs\logs\save_predictions_baselines_run.log and ..\outputs\results\predictions_baselines.csv
pause
