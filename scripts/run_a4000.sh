#!/bin/bash
# Run this INSIDE a JupyterHub Terminal tab (Launcher -> "+" -> Terminal), not a notebook cell.
#
# This machine has exactly 1 GPU (the A4000), so no CUDA_VISIBLE_DEVICES juggling is needed here
# -- torch.cuda.is_available() / nvidia-smi will just see the one card.
#
# Usage:
#   cd ~/milk10k_project/scripts   (adjust if you placed the project elsewhere on JupyterHub)
#   chmod +x run_a4000.sh          (first time only)
#   ./run_a4000.sh
#
# Runs detached with nohup so a dropped browser tab / websocket hiccup doesn't kill training --
# reconnect any time and check progress with:
#   tail -f ../outputs/logs/a4000_run.log
# If the process really did get killed, just re-run this script -- train.py resumes each
# individual run from its last checkpoint automatically (see --no_resume if you ever want a
# clean restart instead).

set -e
cd "$(dirname "$0")/../src"

# First-time environment setup (safe to leave in -- skips if venv already exists):
if [ ! -d "../venv" ]; then
  echo "Creating venv..."
  python3 -m venv ../venv
  source ../venv/bin/activate
  pip install -r ../requirements.txt
else
  source ../venv/bin/activate
fi

mkdir -p ../outputs/logs
nohup python run_all.py --group a4000 --device cuda:0 \
  > ../outputs/logs/a4000_run.log 2>&1 &

echo "Launched in background (PID $!). Log: ../outputs/logs/a4000_run.log"
echo "Check progress any time with: tail -f ../outputs/logs/a4000_run.log"
