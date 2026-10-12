"""
Driver script for the full experimental matrix (CLAUDE.md Phase 3-6), split across the 3 GPUs
available for this project:

  - A4000 (16GB, JupyterHub, idle now)      -> the 90 baseline-multimodal runs + 30
                                                reliability_aware runs (120 runs total). These
                                                need two ConvNeXt-Tiny backbones at once — the
                                                A4000's 16GB + real tensor-core AMP speedup make
                                                it the only one of the three that comfortably
                                                handles this load quickly.
  - GTX 1080 #0 (8GB, local GPU0, busy until ~15:00 WIB tomorrow with LUNA16 dev7_local)
                                              -> half of the 15 unimodal runs (single backbone,
                                                 fits 8GB easily; AMP won't speed Pascal up much,
                                                 but memory isn't the bottleneck for these).
  - GTX 1080 #1 (8GB, local GPU1, busy until ~15:30 WIB tomorrow with LUNA16 dev7b_noweight)
                                              -> the other half of the 15 unimodal runs.

Usage — run the matching command on each machine once its GPU is free:

    # On the A4000 JupyterHub terminal (now):
    python run_all.py --group a4000 --device cuda:0

    # On the local lab PC, GPU0, after LUNA16 dev7_local finishes (~15:00 WIB tomorrow):
    python run_all.py --group gpu0_1080 --device cuda:0

    # On the local lab PC, GPU1, after LUNA16 dev7b_noweight finishes (~15:30 WIB tomorrow):
    python run_all.py --group gpu1_1080 --device cuda:1

    # To see the exact commands for a group before committing:
    python run_all.py --group a4000 --dry_run

Each individual training run auto-resumes from its own "_last" checkpoint if interrupted (e.g. a
JupyterHub browser disconnect) — see train.py. Re-running the same --group command after an
interruption is safe and will skip completed runs' epochs automatically (a run whose "_last"
checkpoint already reached its final epoch will just re-verify and exit quickly rather than
retrain) and resume any partially-finished run from where it stopped.

Run training on a remote/browser-based machine (the A4000 JupyterHub) inside a terminal with
nohup/tmux/screen, NOT a live notebook cell — a closed browser tab can kill a plain foreground
notebook process, but train.py's resume logic means even that is recoverable, just wasteful.
"""
import argparse
import subprocess
import sys

SEVEN_CONFIGS = ["C", "D", "M", "C,D", "C,M", "D,M", "C,D,M"]
SEEDS = [0, 1, 2, 3, 4]
ABLATION_STAGES = ["A1", "A2", "A3", "A4", "A5", "Full"]

# Use the exact same interpreter that is running this script (sys.executable) rather than the
# bare string "python". On Windows in particular, a plain "python" in a subprocess call can
# resolve to a *different* interpreter than the one currently running (e.g. a system install or
# the WindowsApps python.exe stub) even inside an activated venv shell, depending on how PATH is
# inherited -- this caused every train.py subprocess to silently run outside the venv (missing
# pyyaml and friends) despite `pip install` succeeding in the venv. sys.executable is the one
# path that is guaranteed to match the venv actually running this process.
PYTHON = sys.executable


def unimodal_commands():
    cmds = []
    for m in ["C", "D", "M"]:
        for seed in SEEDS:
            cmds.append([PYTHON, "train.py", "--family", "unimodal", "--modality", m,
                         "--seed", str(seed)])
    return cmds


def baseline_multimodal_commands():
    cmds = []
    for family in ["naive_concat", "late_fusion", "gated_fusion"]:
        for modality in SEVEN_CONFIGS:
            if len(modality.split(",")) == 1:
                continue  # single-modality configs are covered by the unimodal family (B1)
            for seed in SEEDS:
                cmds.append([PYTHON, "train.py", "--family", family, "--modality", modality,
                             "--seed", str(seed)])
    return cmds


def reliability_aware_commands():
    """One training run per (seed, ablation_stage); modality dropout inside training covers all
    7 configs, so evaluation (not training) sweeps the 7 configs — see evaluate.py."""
    cmds = []
    for seed in SEEDS:
        for stage in ABLATION_STAGES:
            cmds.append([PYTHON, "train.py", "--family", "reliability_aware",
                         "--modality", "C,D,M", "--seed", str(seed), "--ablation", stage])
    return cmds


def split_unimodal_across_two_gpus():
    """Interleave the 15 unimodal runs round-robin across the two 1080s so each gets a mix of
    modalities/seeds rather than e.g. all of C going to gpu0 and all of M to gpu1 (which would
    make the two machines finish at different times for no good reason)."""
    all_cmds = unimodal_commands()
    gpu0 = all_cmds[0::2]  # 8 runs
    gpu1 = all_cmds[1::2]  # 7 runs
    return gpu0, gpu1


# GTX1080 overflow groups (added 15 Aug, after both 1080s finished the 15 unimodal runs above and
# sat idle while A4000 alone works through the 90-run a4000 group).
#
# a4000's queue is baseline_multimodal_commands() (60 runs) THEN reliability_aware_commands()
# (30 runs), consumed strictly in that order by a single already-running process -- so
# reliability_aware won't be reached for days. These two groups let each idle 1080 pick up half
# of that same 30-run reliability_aware slice RIGHT NOW, in parallel with A4000, with zero risk of
# two machines training the exact same (seed, stage) run at once (A4000 simply hasn't gotten
# there yet). If A4000 nonetheless reaches a (seed, stage) a 1080 already finished, train.py's
# resume logic sees that run's checkpoint already at its final epoch and just re-verifies +
# exits -- it will NOT silently retrain and overwrite it under a different batch_size.
#
# Uses configs/config_gpu1080_reliability.yaml (batch_size=16, everything else identical to
# config.yaml) instead of the default config -- see that file's header comment for why, and for
# the methodological-consistency caveat this introduces (flag in the paper if it matters).
#
# STATUS as of 17 Aug: confirmed working end-to-end, no OOM at batch_size=16. GPU0 finished all
# 15/15 of its half already; GPU1 is finishing its last few runs.
def split_reliability_aware_across_two_gpus():
    all_cmds = reliability_aware_commands()
    for c in all_cmds:
        c += ["--config", "../configs/config_gpu1080_reliability.yaml"]
    gpu0 = all_cmds[0::2]  # 15 runs
    gpu1 = all_cmds[1::2]  # 15 runs
    return gpu0, gpu1


# GTX1080 overflow for baseline_multimodal (added 17 Aug, once GPU0 freed up from the
# reliability_aware overflow above and GPU1 was about to). UNLIKE the reliability_aware overflow
# above, this is NOT automatically collision-free: A4000's queue is
# baseline_multimodal_commands() (60 runs) FIRST, THEN reliability_aware_commands() -- meaning
# A4000 is (or was, as of the last time it was reachable -- it has been unreachable/502 since 15
# Aug) actively partway THROUGH this exact 60-run list, not sitting idle before it the way it was
# for reliability_aware. If a 1080 and A4000 both start training the *same* (family, modality,
# seed) at the same time, they will both write to the same checkpoint file concurrently -- a real
# corruption risk, not just wasted compute.
#
# Safety mechanism: BASELINE_MULTIMODAL_SKIP_FIRST_N below MUST be set (not None), OR
# --baseline_skip_first_n passed on the command line, before either gpu0_1080_base /
# gpu1_1080_base group is run. Before setting it: reconnect to the A4000 JupyterHub terminal, run
# `grep -c "Done. Best" outputs/logs/a4000_run.log` to get its current completed-run count N, then
# use N + 2 (one extra run of buffer beyond N, in case A4000's (N+1)-th run is still in flight and
# not yet reflected as "Done" in the log). The two 1080s then split
# baseline_multimodal_commands()[N+2:] between them and work FORWARD from there, while A4000 keeps
# consuming the list from wherever it actually is, also forward. Every run below index N+2 is left
# to A4000 alone. Once 1080s finish their slice, if A4000 (much slower on this queue) eventually
# reaches those same indices, train.py's resume-skip logic makes it a fast no-op on an
# already-finished run -- same safety property as the reliability_aware overflow.
#
# Uses configs/config_gpu1080_baseline.yaml (batch_size=16) -- baseline_multimodal has no aux
# reliability heads so it needs less VRAM than reliability_aware did, but batch_size=16 is kept
# for consistency with the already-validated config rather than risking a fresh untested batch
# size on a family we haven't run on a 1080 before.
BASELINE_MULTIMODAL_SKIP_FIRST_N = None  # <-- SET THIS (or pass --baseline_skip_first_n) before running


def split_baseline_multimodal_across_two_gpus(skip_first_n=None, config_path="../configs/config_gpu1080_baseline.yaml"):
    n = skip_first_n if skip_first_n is not None else BASELINE_MULTIMODAL_SKIP_FIRST_N
    if n is None:
        raise SystemExit(
            "baseline_skip_first_n is not set.\n"
            "This MUST be set to (A4000's confirmed completed-run count + 2) before running the\n"
            "gpu0_1080_base / gpu1_1080_base groups, to avoid a 1080 and A4000 concurrently\n"
            "training -- and corrupting -- the same checkpoint file. Pass --baseline_skip_first_n N\n"
            "on the command line, or edit BASELINE_MULTIMODAL_SKIP_FIRST_N in this file. See the\n"
            "comment above this function for the exact steps. Do not guess a number without\n"
            "checking the A4000 log first."
        )
    all_cmds = baseline_multimodal_commands()
    remaining = all_cmds[n:]
    if config_path is not None:
        for c in remaining:
            c += ["--config", config_path]
    gpu0 = remaining[0::2]
    gpu1 = remaining[1::2]
    return gpu0, gpu1


GROUPS = {
    "a4000": lambda: baseline_multimodal_commands() + reliability_aware_commands(),
    "gpu0_1080": lambda: split_unimodal_across_two_gpus()[0],
    "gpu1_1080": lambda: split_unimodal_across_two_gpus()[1],
    "gpu0_1080_rel": lambda: split_reliability_aware_across_two_gpus()[0],
    "gpu1_1080_rel": lambda: split_reliability_aware_across_two_gpus()[1],
    "gpu0_1080_base": lambda skip_n: split_baseline_multimodal_across_two_gpus(skip_n)[0],
    "gpu1_1080_base": lambda skip_n: split_baseline_multimodal_across_two_gpus(skip_n)[1],
    # Added 21 Aug: A4000 crashed mid-queue during the 15-19 Aug outage (10/60 baseline_multimodal
    # runs done: naive_concat C+D and C+M, all 5 seeds each), and separately GPU1 on the local
    # 1080 PC hit a CUDA "invalid device ordinal" error on every one of its 30 runs (0 actual
    # training done there). Rather than fix GPU1 locally, route what would have been GPU1's half
    # of the 60-run baseline_multimodal split to the now-reachable A4000 instead -- GPU0 (1080)
    # keeps its half. config_path=None here (unlike gpu0_1080_base/gpu1_1080_base) so these commands
    # do NOT get "--config config_gpu1080_baseline.yaml" appended -- A4000 should keep using its
    # own default config.yaml (larger batch size, more VRAM) for consistency with the 10 runs it
    # already completed under that config; mixing batch sizes within the same baseline family
    # would be a methodological inconsistency worth avoiding, not just a convenience choice.
    # skip_first_n=0 reproduces the exact same 30-command gpu1 list gpu1_1080_base would have used
    # (this is about WHICH config gets appended, not which commands are selected) -- some of these
    # 30 will already be done (checkpoint already at final epoch on A4000's own disk from its
    # earlier crashed run), and train.py's resume logic will just fast-skip those.
    "a4000_gpu1_base": lambda: split_baseline_multimodal_across_two_gpus(skip_first_n=0, config_path=None)[1],
    "all": lambda: unimodal_commands() + baseline_multimodal_commands() + reliability_aware_commands(),
}

# Groups that need the skip_n argument threaded through (the rest ignore it).
GROUPS_NEEDING_SKIP_N = {"gpu0_1080_base", "gpu1_1080_base"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=list(GROUPS.keys()),
                     help="which machine/GPU's slice of the experiment matrix to run")
    ap.add_argument("--device", default="auto",
                     help="torch device to pin every run in this group to, e.g. cuda:0, cuda:1")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--baseline_skip_first_n", type=int, default=None,
                     help="required for gpu0_1080_base/gpu1_1080_base -- see run_all.py comment "
                          "above split_baseline_multimodal_across_two_gpus() for how to compute it")
    args = ap.parse_args()

    if args.group in GROUPS_NEEDING_SKIP_N:
        cmds = GROUPS[args.group](args.baseline_skip_first_n)
    else:
        cmds = GROUPS[args.group]()
    for c in cmds:
        c += ["--device", args.device]

    print(f"Using python interpreter: {PYTHON}", flush=True)
    print(f"Group '{args.group}': {len(cmds)} training runs, device={args.device}", flush=True)
    if args.dry_run:
        for c in cmds:
            print(" ".join(c))
        return

    for i, cmd in enumerate(cmds, 1):
        # flush=True so this header actually lands in the log file *before* the subprocess's own
        # (unbuffered) output, instead of getting stdout-buffered and dumped all at once at exit
        # -- that buffering previously made the log look like all runs failed silently with no
        # headers in between, which was confusing to read.
        print(f"\n=== [{args.group}] Run {i}/{len(cmds)}: {' '.join(cmd)} ===", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Run failed (exit {result.returncode}): {' '.join(cmd)}", file=sys.stderr, flush=True)
            # continue rather than abort the whole group on one failed run
            continue


if __name__ == "__main__":
    main()
