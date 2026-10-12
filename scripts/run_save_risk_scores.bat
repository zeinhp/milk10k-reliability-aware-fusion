@echo off
REM Dumps per-lesion entropy/disagreement/correctness for the Full-stage reliability_aware model
REM (all 7 configs x 5 seeds = 35 cells) -- needed for the paired bootstrap significance test of
REM whether "entropy+disagreement combined" risk actually beats "entropy alone" for error
REM detection, or if that's within noise. Small (35 cells, similar cost to evaluate_baselines).

cd /d "%~dp0..\src"
call ..\venv\Scripts\activate.bat

if not exist "..\outputs\logs" mkdir "..\outputs\logs"

echo Starting per-lesion risk score dump on cuda:0...
python save_risk_scores.py --device cuda:0 >> ..\outputs\logs\save_risk_scores_run.log 2>&1

echo Done. Check ..\outputs\logs\save_risk_scores_run.log and ..\outputs\results\risk_scores_full_stage.csv
pause
