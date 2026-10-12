@echo off
REM Phase 8 analysis: bootstrap CI + subgroup analysis. CPU-only, fast (no GPU/checkpoint access).
REM
REM PREREQUISITE: run BOTH run_gpu0_phase8.bat and run_gpu1_phase8.bat to completion first --
REM this reads predictions_full_stage_gpu0.csv and predictions_full_stage_gpu1.csv, which those
REM scripts' save_predictions.py step produces. Running this before both finish will just error
REM out (missing files) or use incomplete data -- check the outputs\logs\save_predictions_gpu*
REM logs say "Done" before running this.

cd /d "%~dp0..\src"
call ..\venv\Scripts\activate.bat

if not exist "..\outputs\logs" mkdir "..\outputs\logs"
if not exist "..\outputs\results" mkdir "..\outputs\results"

echo [1/2] Bootstrap CI (B=2000, lesion-level, 95%%)...
python bootstrap_ci.py --n_boot 2000 > ..\outputs\logs\bootstrap_ci_run.log 2>&1

echo [2/2] Subgroup analysis (sex, age band, skin tone)...
python subgroup_analysis.py > ..\outputs\logs\subgroup_analysis_run.log 2>&1

echo Done. Results:
echo   ..\outputs\results\bootstrap_ci_full_stage_summary.md
echo   ..\outputs\results\subgroup_analysis_full_stage_summary.md
pause
