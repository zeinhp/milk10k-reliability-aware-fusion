@echo off
REM Extended contradiction stress test -- ALL 6 ablation stages (A1..Full), not just Full.
REM Isolates whether reliability weighting (A2+) actually reduces damage from a contradictory
REM modality, vs. Full alone which we already found is NOT robust to D contradiction (-0.35 F1).
REM Read-only inference, doesn't touch A4000. GPU0 half: seeds 0, 2, 4 (90 of 180 total cells).

cd /d "%~dp0..\src"
call ..\venv\Scripts\activate.bat

if not exist "..\outputs\logs" mkdir "..\outputs\logs"
if not exist "..\outputs\results" mkdir "..\outputs\results"

echo Running extended stress test (GPU0, seeds 0,2,4, all 6 stages)...
python stress_test.py --device cuda:0 --seeds 0,2,4 --csv_suffix _gpu0 > ..\outputs\logs\stress_test_all_stages_gpu0_run.log 2>&1

echo Done. Check ..\outputs\logs\stress_test_all_stages_gpu0_run.log
echo Results: ..\outputs\results\stress_test_contradiction_all_stages_gpu0.csv
pause
