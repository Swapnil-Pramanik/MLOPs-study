import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ------------------------------------------------------------------ #
#  EXPERIMENT RESULT — one run's raw observations                      #
# ------------------------------------------------------------------ #

@dataclass
class ExperimentResult:
    """
    Raw observations from a single fault injection run.
    Filled in by the pipeline's result logger after each experiment.

    Fields
    ------
    pipeline_id     : "P1", "P2", "P3", "P4"
    fault_type      : e.g. "statistical_drift", "endpoint_kill"
    severity        : 0.1, 0.3, or 0.5
    trial           : trial number (0-9)

    ttd             : Time-to-Detect (seconds). How long until fault was flagged.
                      Set to ttd_max if never detected.
    ttr             : Time-to-Remediate (seconds). How long from detection to
                      recovery. Set to sla_budget if never remediated.
    baseline_rmse   : Model RMSE before fault injection (healthy state)
    post_fault_rmse : Model RMSE immediately after fault injection
    post_heal_rmse  : Model RMSE after pipeline attempted healing
    healed          : True if post_heal_rmse within epsilon of baseline_rmse
    remediation_cost: Relative cost of healing (0-1). e.g. full retrain=1.0,
                      simple rollback=0.2, no action=0.0
    post_recovery_variance : Variance of RMSE over 5 batches after healing.
                             Low = stable recovery, High = oscillating.
    false_positives : Number of false drift alarms during the observation window
    total_detections: Total number of detection events fired
    """
    pipeline_id:             str
    fault_type:              str
    severity:                float
    trial:                   int

    ttd:                     float   # seconds
    ttr:                     float   # seconds
    baseline_rmse:           float
    post_fault_rmse:         float
    post_heal_rmse:          float
    healed:                  bool

    remediation_cost:        float   = 0.0   # 0.0–1.0
    post_recovery_variance:  float   = 0.0
    false_positives:         int     = 0
    total_detections:        int     = 1

    # constants — can override per experiment
    ttd_max:                 float   = 60.0  # seconds — worst acceptable TTD
    sla_budget:              float   = 120.0 # seconds — max TTR before SLA breach


# ------------------------------------------------------------------ #
#  SHS DIMENSIONS                                                      #
# ------------------------------------------------------------------ #

@dataclass
class SHSDimensions:
    """
    Stores the four SHS sub-scores before weighting.
    Useful for radar chart plotting and per-dimension analysis.
    """
    D: float  # Detection Efficacy     (0–1)
    R: float  # Response Efficiency    (0–1)
    C: float  # Fault Coverage         (0–1)
    S: float  # Post-Healing Stability (0–1)
    SHS: float  # Weighted composite   (0–1)


# ------------------------------------------------------------------ #
#  CORE SHS COMPUTATION                                                #
# ------------------------------------------------------------------ #

def compute_D(result: ExperimentResult) -> float:
    """
    Detection Efficacy.

    Measures how quickly AND accurately the pipeline detected the fault.
    Penalizes both slow detection and false positives.

    Formula:
        raw_D     = 1 - (TTD / TTD_max)          [speed term]
        precision = 1 / (1 + false_positives)     [accuracy term]
        D         = raw_D * precision

    Range: 0 (never detected / many false alarms) → 1 (instant, no false alarms)
    """
    if result.ttd >= result.ttd_max:
        raw_d = 0.0  # never detected within acceptable window
    else:
        raw_d = 1.0 - (result.ttd / result.ttd_max)

    precision = 1.0 / (1.0 + result.false_positives)
    D = raw_d * precision
    return float(np.clip(D, 0.0, 1.0))


def compute_R(result: ExperimentResult) -> float:
    """
    Response Efficiency.

    Measures how fast AND cheaply the pipeline healed.
    A pipeline that heals instantly but always does a full retrain
    scores lower than one that heals quickly with a cheap rollback.

    Formula:
        speed_term = 1 - (TTR / SLA_budget)
        cost_term  = 1 / (1 + remediation_cost)
        R          = speed_term * cost_term

    Range: 0 (SLA breached / expensive) → 1 (instant, cheap)
    """
    if result.ttr >= result.sla_budget:
        speed_term = 0.0
    else:
        speed_term = 1.0 - (result.ttr / result.sla_budget)

    cost_term = 1.0 / (1.0 + result.remediation_cost)
    R = speed_term * cost_term
    return float(np.clip(R, 0.0, 1.0))


def compute_S(result: ExperimentResult, epsilon: float = 0.05) -> float:
    """
    Post-Healing Stability.

    Measures whether the pipeline truly returned to baseline,
    and whether it stayed there (no oscillation).

    Formula:
        recovery_quality = 1 - |post_heal_rmse - baseline_rmse| / baseline_rmse
        stability_term   = 1 - normalized(post_recovery_variance)
        S = recovery_quality * stability_term   if healed
        S = 0                                   if not healed

    Range: 0 (never healed or oscillating) → 1 (perfect stable recovery)
    """
    if not result.healed:
        return 0.0

    # How close did we get back to baseline?
    relative_error = abs(result.post_heal_rmse - result.baseline_rmse)
    relative_error /= (result.baseline_rmse + 1e-9)
    recovery_quality = max(0.0, 1.0 - relative_error)

    # How stable is post-recovery performance?
    # Normalize variance by baseline_rmse^2 to make it scale-invariant
    normalized_var = result.post_recovery_variance / (result.baseline_rmse ** 2 + 1e-9)
    stability_term = max(0.0, 1.0 - normalized_var)

    S = recovery_quality * stability_term
    return float(np.clip(S, 0.0, 1.0))


def compute_single_SHS(result: ExperimentResult,
                        C: float,
                        weights: tuple = (0.25, 0.30, 0.25, 0.20)) -> SHSDimensions:
    """
    Computes all four SHS dimensions and the composite score
    for a single ExperimentResult.

    Parameters
    ----------
    result  : ExperimentResult for a single trial
    C       : Fault Coverage score for this pipeline (computed across all trials,
              passed in from compute_pipeline_SHS)
    weights : (w1, w2, w3, w4) for (D, R, C, S). Must sum to 1.0.

    Returns
    -------
    SHSDimensions with all sub-scores and composite SHS.
    """
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
    w1, w2, w3, w4 = weights

    D = compute_D(result)
    R = compute_R(result)
    S = compute_S(result)
    SHS = w1 * D + w2 * R + w3 * C + w4 * S

    return SHSDimensions(D=D, R=R, C=C, S=S, SHS=SHS)


# ------------------------------------------------------------------ #
#  PIPELINE-LEVEL AGGREGATION                                          #
# ------------------------------------------------------------------ #

def compute_pipeline_SHS(results: list[ExperimentResult],
                          weights: tuple = (0.25, 0.30, 0.25, 0.20)) -> pd.DataFrame:
    """
    Computes SHS for an entire pipeline across all fault types and trials.

    Steps:
    1. Compute C (fault coverage) = fraction of (fault, severity) combos healed
    2. Compute per-trial D, R, S scores
    3. Return a DataFrame with one row per trial for statistical analysis

    Parameters
    ----------
    results : list of ExperimentResult from all trials for one pipeline
    weights : SHS weight vector

    Returns
    -------
    DataFrame with columns:
        pipeline_id, fault_type, severity, trial, D, R, C, S, SHS
    """
    # Compute C: fault coverage across all unique (fault, severity) combos
    total_combos = len(set((r.fault_type, r.severity) for r in results))
    healed_combos = len(set(
        (r.fault_type, r.severity)
        for r in results
        if r.healed
    ))
    C = healed_combos / total_combos if total_combos > 0 else 0.0

    rows = []
    for result in results:
        dims = compute_single_SHS(result, C=C, weights=weights)
        rows.append({
            "pipeline_id":  result.pipeline_id,
            "fault_type":   result.fault_type,
            "severity":     result.severity,
            "trial":        result.trial,
            "D":            dims.D,
            "R":            dims.R,
            "C":            dims.C,
            "S":            dims.S,
            "SHS":          dims.SHS,
            "healed":       result.healed,
            "ttd":          result.ttd,
            "ttr":          result.ttr,
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
#  SUMMARY STATISTICS                                                  #
# ------------------------------------------------------------------ #

def summarize_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates per-trial SHS DataFrame into summary statistics.
    Reports mean ± 95% CI for each dimension per fault type.

    Input  : output of compute_pipeline_SHS()
    Output : summary DataFrame for paper tables
    """
    from scipy import stats

    def ci95(x):
        if len(x) < 2:
            return 0.0
        return stats.sem(x) * stats.t.ppf(0.975, df=len(x) - 1)

    summary = df.groupby(["pipeline_id", "fault_type", "severity"]).agg(
        D_mean=("D",   "mean"),
        D_ci  =("D",   ci95),
        R_mean=("R",   "mean"),
        R_ci  =("R",   ci95),
        C_mean=("C",   "mean"),
        S_mean=("S",   "mean"),
        S_ci  =("S",   ci95),
        SHS_mean=("SHS", "mean"),
        SHS_ci  =("SHS", ci95),
        heal_rate=("healed", "mean"),
        n_trials =("trial",  "count"),
    ).reset_index()

    return summary


# ------------------------------------------------------------------ #
#  SENSITIVITY ANALYSIS                                                #
# ------------------------------------------------------------------ #

def sensitivity_analysis(results: list[ExperimentResult],
                          n_perturbations: int = 100,
                          seed: int = 42) -> pd.DataFrame:
    """
    Tests SHS rank stability under random weight perturbations.
    Key validation: if pipeline rankings stay consistent across
    perturbed weight vectors, the metric is not gaming-sensitive.

    Generates n_perturbations random weight vectors from a Dirichlet
    distribution (ensures they sum to 1), computes SHS for each,
    and returns rank correlation statistics.

    This is a key result for the paper — demonstrates metric robustness.
    """
    rng = np.random.default_rng(seed)
    pipeline_ids = list(set(r.pipeline_id for r in results))
    results_by_pipeline = {
        pid: [r for r in results if r.pipeline_id == pid]
        for pid in pipeline_ids
    }

    rank_records = []
    for i in range(n_perturbations):
        # Random weights summing to 1 via Dirichlet
        w = rng.dirichlet(alpha=[1, 1, 1, 1])
        w_tuple = tuple(w)

        scores = {}
        for pid, pid_results in results_by_pipeline.items():
            df = compute_pipeline_SHS(pid_results, weights=w_tuple)
            scores[pid] = df["SHS"].mean()

        ranked = sorted(scores, key=scores.get, reverse=True)
        rank_records.append({
            "w1": w[0], "w2": w[1], "w3": w[2], "w4": w[3],
            **{f"rank_{pid}": ranked.index(pid) + 1 for pid in pipeline_ids},
            **{f"SHS_{pid}": scores[pid] for pid in pipeline_ids},
        })

    return pd.DataFrame(rank_records)


# ------------------------------------------------------------------ #
#  SELF TEST
#  python shs_metric.py
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=" * 55)
    print("SHS Metric Self-Test")
    print("=" * 55)

    # Create 40 dummy results — 4 pipelines × 2 faults × 2 severities × 2 trials
    dummy_results = []
    pipelines = ["P1", "P2", "P3", "P4"]
    faults = ["statistical_drift", "endpoint_kill"]
    severities = [0.1, 0.5]

    rng = np.random.default_rng(99)

    # Simulate P1 being best at drift, P2 best at endpoint faults
    perf_profile = {
        "P1": {"statistical_drift": 0.8, "endpoint_kill": 0.4},
        "P2": {"statistical_drift": 0.5, "endpoint_kill": 0.9},
        "P3": {"statistical_drift": 0.6, "endpoint_kill": 0.7},
        "P4": {"statistical_drift": 0.7, "endpoint_kill": 0.6},
    }

    for pid in pipelines:
        for fault in faults:
            for sev in severities:
                for trial in range(5):
                    base_perf = perf_profile[pid][fault]
                    noise = rng.normal(0, 0.05)
                    healed = rng.random() < (base_perf - sev * 0.2)

                    r = ExperimentResult(
                        pipeline_id=pid,
                        fault_type=fault,
                        severity=sev,
                        trial=trial,
                        ttd=rng.uniform(2, 60) * (1 - base_perf + 0.1),
                        ttr=rng.uniform(5, 120) * (1 - base_perf + 0.1),
                        baseline_rmse=4.5,
                        post_fault_rmse=4.5 * (1 + sev * 2),
                        post_heal_rmse=4.5 * (1 + (0.05 if healed else sev)),
                        healed=healed,
                        remediation_cost=rng.uniform(0.1, 0.8),
                        post_recovery_variance=rng.uniform(0.01, 0.5),
                        false_positives=rng.integers(0, 3),
                        total_detections=1,
                    )
                    dummy_results.append(r)

    # Test per-pipeline SHS
    print("\n--- Pipeline SHS Scores ---")
    all_dfs = []
    for pid in pipelines:
        pid_results = [r for r in dummy_results if r.pipeline_id == pid]
        df = compute_pipeline_SHS(pid_results)
        mean_shs = df["SHS"].mean()
        print(f"  {pid}: SHS = {mean_shs:.3f}  "
              f"(D={df['D'].mean():.2f}, R={df['R'].mean():.2f}, "
              f"C={df['C'].mean():.2f}, S={df['S'].mean():.2f})")
        all_dfs.append(df)

    combined = pd.concat(all_dfs)
    assert "SHS" in combined.columns, "SHS column missing"
    assert combined["SHS"].between(0, 1).all(), "SHS out of [0,1] range"
    print("\n✅ SHS values all in [0, 1]: passed")

    # Test summary
    summary = summarize_pipeline(combined)
    assert "SHS_mean" in summary.columns, "Summary missing SHS_mean"
    print("✅ Summary statistics: passed")

    # Test sensitivity analysis
    print("\n--- Sensitivity Analysis (100 weight perturbations) ---")
    sens_df = sensitivity_analysis(dummy_results, n_perturbations=100)
    rank_cols = [c for c in sens_df.columns if c.startswith("rank_")]
    rank_stability = sens_df[rank_cols].std().mean()
    print(f"  Mean rank std across perturbations: {rank_stability:.3f}")
    print(f"  (Lower = more stable rankings = more robust metric)")
    print("✅ Sensitivity analysis: passed")

    print("\n✅ All SHS metric tests passed. Ready for pipeline integration.")