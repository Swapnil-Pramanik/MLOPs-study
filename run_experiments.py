"""
SH-Bench Master Experiment Runner
==================================
Run this on Legion Pro 7 on Day 2.
Executes all pipeline x fault x severity x trial combinations,
logs every result immediately to results/raw_results.csv.

Usage:
    python run_experiments.py

Resume after crash:
    python run_experiments.py --resume

Dry run (verify setup without running):
    python run_experiments.py --dry-run

Single pipeline test:
    python run_experiments.py --pipeline P1 --trials 2
"""

import argparse
import os
import sys
import time
import traceback
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product

sys.path.append(os.path.dirname(__file__))

from data.data_loader import load_taxi, FEATURE_COLS, TARGET_COL
from fault_injector.fault_injector import FaultInjector
from evaluator.result_logger import ResultLogger
from evaluator.shs_metric import ExperimentResult

from pipelines.p1_drift_retrain.p1_pipeline    import P1Pipeline
from pipelines.p2_circuit_breaker.p2_pipeline  import P2Pipeline
from pipelines.p3_rl_autoscaler.p3_pipeline    import P3Pipeline, PPO_TIMESTEPS
from pipelines.p4_causal_rca.p4_pipeline       import P4Pipeline


# ------------------------------------------------------------------ #
#  EXPERIMENT MATRIX                                                   #
# ------------------------------------------------------------------ #

FAULT_TYPES = [
    "statistical_drift",
    "label_poison",
    "batch_corruption",
    "concept_drift",
    "endpoint_kill",
    "schema_drift",
    "memory_pressure",
    "compound_fault",
]

SEVERITIES  = [0.1, 0.3, 0.5]
N_TRIALS    = 10

DATA_PATH_2019 = os.path.join("data", "yellow_tripdata_2019-01.csv")
DATA_PATH_2020 = os.path.join("data", "yellow_tripdata_2020-01.csv")
SAMPLE_SIZE    = 50000
TRAIN_SIZE     = 40000
AGENT_PATH     = os.path.join("models", "p3_agent")


# ------------------------------------------------------------------ #
#  PIPELINE FACTORY                                                    #
# ------------------------------------------------------------------ #

def build_pipelines(df_train: pd.DataFrame,
                    df_reference: pd.DataFrame,
                    p3_agent_path: str) -> dict:
    """
    Instantiates and trains all 4 pipelines.
    P3 agent is trained once and reused across all trials.
    Returns dict of pipeline_id -> pipeline instance.
    """
    pipelines = {}

    print("\n[Setup] Training P1...")
    p1 = P1Pipeline(df_reference=df_reference)
    p1.initial_train(df_train)
    pipelines["P1"] = p1

    print("[Setup] Training P2...")
    p2 = P2Pipeline(df_reference=df_reference)
    p2.initial_train(df_train)
    pipelines["P2"] = p2

    print("[Setup] Training P3 model...")
    p3 = P3Pipeline(df_reference=df_reference)
    p3.initial_train(df_train)

    # Load or train PPO agent
    agent_file = p3_agent_path + ".zip"
    if os.path.exists(agent_file):
        print(f"[Setup] Loading existing P3 agent from {agent_file}")
        p3.load_agent(p3_agent_path)
    else:
        print(f"[Setup] Training P3 PPO agent "
              f"({PPO_TIMESTEPS} timesteps)...")
        p3.train_agent(total_timesteps=PPO_TIMESTEPS)
        p3.save_agent(p3_agent_path)
        print(f"[Setup] P3 agent saved to {agent_file}")
    pipelines["P3"] = p3

    print("[Setup] Training P4...")
    p4 = P4Pipeline(df_reference=df_reference)
    p4.initial_train(df_train)
    pipelines["P4"] = p4

    print("[Setup] All pipelines ready.\n")
    return pipelines


def reset_pipeline(pipeline_id: str,
                   df_train: pd.DataFrame,
                   df_reference: pd.DataFrame,
                   p3_instance: object = None) -> object:
    """
    Returns a fresh pipeline instance for each trial.
    Prevents state leakage between trials.
    P3 reuses the pre-trained agent — no retraining.
    """
    if pipeline_id == "P1":
        p = P1Pipeline(df_reference=df_reference)
        p.initial_train(df_train)
        return p
    elif pipeline_id == "P2":
        p = P2Pipeline(df_reference=df_reference)
        p.initial_train(df_train)
        return p
    elif pipeline_id == "P3":
        p = P3Pipeline(df_reference=df_reference)
        p.initial_train(df_train)
        p.agent = p3_instance.agent  # reuse trained agent
        return p
    elif pipeline_id == "P4":
        p = P4Pipeline(df_reference=df_reference)
        p.initial_train(df_train)
        return p
    raise ValueError(f"Unknown pipeline: {pipeline_id}")


# ------------------------------------------------------------------ #
#  ALREADY-RUN CHECKER                                                 #
# ------------------------------------------------------------------ #

def get_completed_runs(raw_csv_path: str) -> set:
    """
    Reads existing raw_results.csv and returns set of
    (pipeline_id, fault_type, severity, trial) tuples
    that have already been completed.
    Used for --resume mode.
    """
    if not os.path.exists(raw_csv_path):
        return set()
    df = pd.read_csv(raw_csv_path)
    completed = set(
        zip(df["pipeline_id"],
            df["fault_type"],
            df["severity"].astype(float),
            df["trial"].astype(int))
    )
    print(f"[Resume] Found {len(completed)} completed runs.")
    return completed


# ------------------------------------------------------------------ #
#  EXPERIMENT LOOP                                                     #
# ------------------------------------------------------------------ #

def run_all_experiments(pipelines: list,
                        df_train: pd.DataFrame,
                        df_stream: pd.DataFrame,
                        df_reference: pd.DataFrame,
                        logger: ResultLogger,
                        resume: bool = False,
                        dry_run: bool = False) -> None:
    """
    Main experiment loop.
    Iterates over all pipeline x fault x severity x trial combinations.
    """
    p3_instance = None
    for pid, _ in pipelines:
        if pid == "P3":
            p3_instance = _  # keep P3 reference for agent reuse
            break

    # Full combination list
    combos = list(product(
        [pid for pid, _ in pipelines],
        FAULT_TYPES,
        SEVERITIES,
        range(N_TRIALS)
    ))
    total = len(combos)

    # Resume: skip already-completed runs
    completed = set()
    if resume:
        completed = get_completed_runs(logger.raw_csv_path)
        remaining = [(p, f, s, t) for p, f, s, t in combos
                     if (p, f, s, t) not in completed]
        print(f"[Resume] {len(completed)} done, "
              f"{len(remaining)} remaining.")
        combos = remaining

    if dry_run:
        print(f"\n[DryRun] Would run {len(combos)} experiments:")
        for pid, fault, sev, trial in combos[:10]:
            print(f"  {pid} | {fault} | sev={sev} | trial={trial}")
        if len(combos) > 10:
            print(f"  ... and {len(combos)-10} more")
        print(f"\n[DryRun] Total: {len(combos)} runs")
        print(f"[DryRun] Estimated time: "
              f"~{len(combos) * 45 / 60:.0f} minutes")
        return

    print(f"\n{'='*55}")
    print(f"Starting experiment loop: {len(combos)} runs")
    print(f"Estimated time: ~{len(combos) * 45 / 60:.0f} minutes")
    print(f"{'='*55}\n")

    injector    = FaultInjector(random_seed=42)
    pipeline_dict = {pid: p for pid, p in pipelines}

    # Progress bar
    pbar = tqdm(combos, desc="Experiments", unit="run",
                ncols=80, colour="green")

    failed_runs = []

    for pid, fault_type, severity, trial in pbar:
        pbar.set_description(
            f"{pid}|{fault_type[:12]}|s={severity}|t={trial}")

        # Skip infeasible combos
        # weight_corruption needs a fitted sklearn model —
        # handled inside pipeline, but log skip for transparency
        if fault_type == "weight_corruption":
            # Only P1 and P4 support this via their model objects
            if pid in ["P2", "P3"]:
                continue

        try:
            # Fresh pipeline instance per trial — no state leakage
            pipeline = reset_pipeline(
                pid, df_train, df_reference,
                p3_instance=pipeline_dict.get("P3"))

            # Run experiment
            result = pipeline.run_experiment(
                df_stream  = df_stream,
                fault_type = fault_type,
                severity   = severity,
                trial      = trial,
                injector   = injector
            )

            # Log immediately
            logger.log(result)

        except KeyboardInterrupt:
            print("\n[Runner] Interrupted by user. "
                  "Results saved so far are safe.")
            print(f"[Runner] Completed {logger.run_count} runs. "
                  "Run with --resume to continue.")
            sys.exit(0)

        except Exception as e:
            print(f"\n[Runner] ERROR: {pid}|{fault_type}|"
                  f"sev={severity}|trial={trial}")
            print(f"  {type(e).__name__}: {e}")
            traceback.print_exc()
            failed_runs.append((pid, fault_type, severity, trial))
            # Continue with next run — don't stop the loop
            continue

        # Progress update every 20 runs
        if logger.run_count % 20 == 0:
            logger.progress(total_expected=total)

    print(f"\n{'='*55}")
    print(f"Experiment loop complete.")
    print(f"Total runs logged: {logger.run_count}")
    if failed_runs:
        print(f"Failed runs ({len(failed_runs)}):")
        for r in failed_runs:
            print(f"  {r}")
    print(f"{'='*55}\n")


# ------------------------------------------------------------------ #
#  MAIN                                                                #
# ------------------------------------------------------------------ #

def main():
    global FAULT_TYPES, N_TRIALS
    parser = argparse.ArgumentParser(
        description="SH-Bench Experiment Runner")
    parser.add_argument("--resume",   action="store_true",
                        help="Skip already-completed runs")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show what would run without running")
    parser.add_argument("--pipeline", type=str, default=None,
                        choices=["P1", "P2", "P3", "P4"],
                        help="Run only one pipeline")
    parser.add_argument("--trials",   type=int, default=N_TRIALS,
                        help=f"Trials per combo (default {N_TRIALS})")
    parser.add_argument("--faults",   type=str, default=None,
                        nargs="+", choices=FAULT_TYPES,
                        help="Run only specific fault types")
    args = parser.parse_args()

    print("=" * 55)
    print("SH-Bench Experiment Runner")
    print("=" * 55)

    # ── Load data ─────────────────────────────────────────────────
    print("\n[Setup] Loading datasets...")
    df_full = load_taxi(DATA_PATH_2019, sample_size=SAMPLE_SIZE)
    df_train     = df_full.iloc[:TRAIN_SIZE].reset_index(drop=True)
    df_stream    = df_full.iloc[TRAIN_SIZE:].reset_index(drop=True)
    df_reference = df_train.copy()
    print(f"[Setup] Train: {len(df_train)} rows | "
          f"Stream: {len(df_stream)} rows")

    # ── Build pipelines ───────────────────────────────────────────
    pipeline_ids = ([args.pipeline] if args.pipeline
                    else ["P1", "P2", "P3", "P4"])
    raw_pipelines = build_pipelines(df_train, df_reference, AGENT_PATH)
    pipelines = [(pid, raw_pipelines[pid]) for pid in pipeline_ids]

    # ── Override experiment matrix if args provided ───────────────
    if args.faults:
        FAULT_TYPES = args.faults
    if args.trials:
        N_TRIALS = args.trials

    # ── Initialize logger ─────────────────────────────────────────
    logger = ResultLogger()

    # ── Run experiments ───────────────────────────────────────────
    start_time = time.time()
    run_all_experiments(
        pipelines    = pipelines,
        df_train     = df_train,
        df_stream    = df_stream,
        df_reference = df_reference,
        logger       = logger,
        resume       = args.resume,
        dry_run      = args.dry_run,
    )
    elapsed = time.time() - start_time

    if not args.dry_run:
        # ── Finalize SHS scores ───────────────────────────────────
        print("[Runner] Computing SHS scores for all results...")
        shs_df = logger.finalize()

        print(f"\n[Runner] Total elapsed time: "
              f"{elapsed/60:.1f} minutes")
        print(f"[Runner] Results saved to results/raw_results.csv")
        print(f"[Runner] SHS scores saved to results/shs_results.csv")
        print("\n[Runner] Quick SHS Summary:")
        if shs_df is not None:
            summary = shs_df.groupby("pipeline_id")["SHS"].agg(
                ["mean", "std", "count"])
            print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
