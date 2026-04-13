import os
import csv
import json
import pandas as pd
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluator.shs_metric import ExperimentResult, compute_pipeline_SHS


# ------------------------------------------------------------------ #
#  CONSTANTS                                                           #
# ------------------------------------------------------------------ #

RESULTS_DIR     = os.path.join(os.path.dirname(__file__), '..', 'results')
RAW_RESULTS_CSV = os.path.join(RESULTS_DIR, 'raw_results.csv')
SHS_RESULTS_CSV = os.path.join(RESULTS_DIR, 'shs_results.csv')
RUN_LOG_JSON    = os.path.join(RESULTS_DIR, 'run_log.json')

RAW_CSV_FIELDS = [
    "timestamp", "pipeline_id", "fault_type", "severity", "trial",
    "ttd", "ttr", "baseline_rmse", "post_fault_rmse", "post_heal_rmse",
    "healed", "remediation_cost", "post_recovery_variance",
    "false_positives", "total_detections"
]


# ------------------------------------------------------------------ #
#  LOGGER                                                              #
# ------------------------------------------------------------------ #

class ResultLogger:
    """
    Logs ExperimentResults to CSV after every single run.
    Nothing is held in memory — every result is immediately
    written to disk so no data is lost if the experiment loop crashes.

    Usage:
        logger = ResultLogger()
        result = pipeline.run_experiment(...)
        logger.log(result)

    After all experiments:
        logger.finalize()   # computes SHS for all logged results
    """

    def __init__(self, results_dir: str = RESULTS_DIR):
        self.results_dir     = results_dir
        self.raw_csv_path    = os.path.join(results_dir, 'raw_results.csv')
        self.shs_csv_path    = os.path.join(results_dir, 'shs_results.csv')
        self.run_log_path    = os.path.join(results_dir, 'run_log.json')
        self.run_count       = 0
        self.session_start   = datetime.now().isoformat()

        os.makedirs(results_dir, exist_ok=True)
        self._init_csv()

        print(f"[Logger] Initialized. Writing to {self.raw_csv_path}")

    def _init_csv(self):
        """Creates CSV with headers if it doesn't already exist."""
        if not os.path.exists(self.raw_csv_path):
            with open(self.raw_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=RAW_CSV_FIELDS)
                writer.writeheader()
            print(f"[Logger] Created new results file.")
        else:
            # Count existing rows
            existing = pd.read_csv(self.raw_csv_path)
            self.run_count = len(existing)
            print(f"[Logger] Appending to existing file. "
                  f"{self.run_count} runs already logged.")

    # ── Core logging ──────────────────────────────────────────────

    def log(self, result: ExperimentResult):
        """
        Writes one ExperimentResult to raw_results.csv immediately.
        Call this after every single pipeline.run_experiment() call.
        """
        row = {
            "timestamp":             datetime.now().isoformat(),
            "pipeline_id":           result.pipeline_id,
            "fault_type":            result.fault_type,
            "severity":              result.severity,
            "trial":                 result.trial,
            "ttd":                   round(result.ttd, 4),
            "ttr":                   round(result.ttr, 4),
            "baseline_rmse":         round(result.baseline_rmse, 6),
            "post_fault_rmse":       round(result.post_fault_rmse, 6),
            "post_heal_rmse":        round(result.post_heal_rmse, 6),
            "healed":                int(result.healed),
            "remediation_cost":      round(result.remediation_cost, 4),
            "post_recovery_variance":round(result.post_recovery_variance, 6),
            "false_positives":       result.false_positives,
            "total_detections":      result.total_detections,
        }

        with open(self.raw_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=RAW_CSV_FIELDS)
            writer.writerow(row)

        self.run_count += 1
        self._update_run_log(result)

        print(f"  [Logger] Run #{self.run_count} saved | "
              f"{result.pipeline_id} | {result.fault_type} | "
              f"sev={result.severity} | trial={result.trial} | "
              f"healed={result.healed}")

    def _update_run_log(self, result: ExperimentResult):
        """
        Maintains a lightweight JSON log of run counts per pipeline.
        Useful for monitoring progress during the long experiment loop.
        """
        if os.path.exists(self.run_log_path):
            with open(self.run_log_path, 'r') as f:
                log = json.load(f)
        else:
            log = {
                "session_start": self.session_start,
                "total_runs": 0,
                "by_pipeline": {"P1": 0, "P2": 0, "P3": 0, "P4": 0},
                "by_fault": {}
            }

        log["total_runs"] = self.run_count
        log["by_pipeline"][result.pipeline_id] = \
            log["by_pipeline"].get(result.pipeline_id, 0) + 1
        log["by_fault"][result.fault_type] = \
            log["by_fault"].get(result.fault_type, 0) + 1
        log["last_updated"] = datetime.now().isoformat()

        with open(self.run_log_path, 'w') as f:
            json.dump(log, f, indent=2)

    # ── Progress monitoring ───────────────────────────────────────

    def progress(self, total_expected: int):
        """
        Prints a progress summary.
        Call periodically during the experiment loop to check status.

        Usage:
            logger.progress(total_expected=200)
        """
        if os.path.exists(self.run_log_path):
            with open(self.run_log_path, 'r') as f:
                log = json.load(f)
            pct = (log["total_runs"] / total_expected) * 100
            print(f"\n[Progress] {log['total_runs']}/{total_expected} "
                  f"({pct:.1f}%) complete")
            print(f"  By pipeline: {log['by_pipeline']}")
            print(f"  By fault:    {log['by_fault']}")
            print(f"  Last update: {log.get('last_updated', 'N/A')}\n")

    # ── Finalization ──────────────────────────────────────────────

    def finalize(self, weights: tuple = (0.25, 0.30, 0.25, 0.20)):
        """
        Called after all experiments are done.
        Reads raw_results.csv, computes SHS for all results,
        writes shs_results.csv.

        This is the file you use for all analysis and plotting.
        """
        print(f"\n[Logger] Finalizing {self.run_count} results...")
        raw_df = pd.read_csv(self.raw_csv_path)

        if len(raw_df) == 0:
            print("[Logger] No results to finalize.")
            return None

        # Reconstruct ExperimentResult objects
        all_results = []
        for _, row in raw_df.iterrows():
            r = ExperimentResult(
                pipeline_id             = row["pipeline_id"],
                fault_type              = row["fault_type"],
                severity                = float(row["severity"]),
                trial                   = int(row["trial"]),
                ttd                     = float(row["ttd"]),
                ttr                     = float(row["ttr"]),
                baseline_rmse           = float(row["baseline_rmse"]),
                post_fault_rmse         = float(row["post_fault_rmse"]),
                post_heal_rmse          = float(row["post_heal_rmse"]),
                healed                  = bool(int(row["healed"])),
                remediation_cost        = float(row["remediation_cost"]),
                post_recovery_variance  = float(row["post_recovery_variance"]),
                false_positives         = int(row["false_positives"]),
                total_detections        = int(row["total_detections"]),
            )
            all_results.append(r)

        # Compute SHS per pipeline
        all_shs_dfs = []
        for pid in raw_df["pipeline_id"].unique():
            pid_results = [r for r in all_results if r.pipeline_id == pid]
            shs_df = compute_pipeline_SHS(pid_results, weights=weights)
            all_shs_dfs.append(shs_df)

        combined_shs = pd.concat(all_shs_dfs, ignore_index=True)
        combined_shs.to_csv(self.shs_csv_path, index=False)
        print(f"[Logger] SHS results written to {self.shs_csv_path}")

        # Print quick summary
        print("\n--- SHS Summary ---")
        summary = combined_shs.groupby("pipeline_id")["SHS"].agg(
            ["mean", "std", "count"])
        print(summary.to_string())

        return combined_shs


# ------------------------------------------------------------------ #
#  SELF TEST
#  python evaluator/result_logger.py
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import numpy as np
    from evaluator.shs_metric import ExperimentResult

    print("=" * 55)
    print("ResultLogger Self-Test")
    print("=" * 55)

    # Use a temp directory so we don't pollute real results
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    logger  = ResultLogger(results_dir=tmp_dir)

    rng = np.random.default_rng(42)

    # Simulate 8 experiment results
    for i in range(8):
        r = ExperimentResult(
            pipeline_id             = f"P{(i % 4) + 1}",
            fault_type              = "statistical_drift",
            severity                = 0.3,
            trial                   = i,
            ttd                     = rng.uniform(5, 55),
            ttr                     = rng.uniform(10, 110),
            baseline_rmse           = 4.5,
            post_fault_rmse         = 6.0,
            post_heal_rmse          = rng.uniform(4.5, 5.5),
            healed                  = rng.random() > 0.3,
            remediation_cost        = rng.uniform(0.1, 1.0),
            post_recovery_variance  = rng.uniform(0.01, 0.3),
            false_positives         = rng.integers(0, 3),
            total_detections        = 1,
        )
        logger.log(r)

    logger.progress(total_expected=8)
    shs_df = logger.finalize()

    assert shs_df is not None, "Finalize returned None"
    assert os.path.exists(os.path.join(tmp_dir, 'raw_results.csv'))
    assert os.path.exists(os.path.join(tmp_dir, 'shs_results.csv'))
    assert os.path.exists(os.path.join(tmp_dir, 'run_log.json'))

    print("\n✅ All ResultLogger tests passed.")
    print(f"   raw_results.csv  : {len(pd.read_csv(os.path.join(tmp_dir, 'raw_results.csv')))} rows")
    print(f"   shs_results.csv  : {len(shs_df)} rows")