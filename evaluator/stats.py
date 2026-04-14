import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations


# ------------------------------------------------------------------ #
#  FRIEDMAN TEST                                                       #
# ------------------------------------------------------------------ #

def friedman_test(shs_df: pd.DataFrame) -> dict:
    """
    Non-parametric Friedman test across all pipelines.
    Tests whether SHS score differences between pipelines are
    statistically significant or could be due to random chance.

    H0: All pipelines perform equally (SHS distributions are the same)
    H1: At least one pipeline differs significantly

    Parameters
    ----------
    shs_df : output of compute_pipeline_SHS() or result_logger.finalize()
             Must contain columns: pipeline_id, fault_type, severity, SHS

    Returns
    -------
    dict with:
        statistic : Friedman test statistic
        p_value   : p-value (< 0.05 = reject H0 = rankings are significant)
        significant : bool
        interpretation : plain English result for paper
    """
    pipelines = shs_df["pipeline_id"].unique()

    # Build matrix: rows = (fault, severity) blocks, cols = pipelines
    blocks = shs_df.groupby(["fault_type", "severity"])
    groups = []

    for pid in pipelines:
        pid_scores = []
        for (fault, sev), block in blocks:
            scores = block[block["pipeline_id"] == pid]["SHS"].values
            pid_scores.append(np.mean(scores) if len(scores) > 0 else np.nan)
        groups.append(pid_scores)

    # Remove blocks with NaN
    groups_array = np.array(groups)
    valid_cols = ~np.any(np.isnan(groups_array), axis=0)
    groups_clean = [g[valid_cols] for g in groups_array]

    statistic, p_value = stats.friedmanchisquare(*groups_clean)
    significant = p_value < 0.05

    interpretation = (
        f"Friedman test (χ²={statistic:.3f}, p={p_value:.4f}): "
        + ("Significant differences exist between pipelines (p < 0.05). "
           "SHS rankings are not due to chance."
           if significant else
           "No significant difference detected (p ≥ 0.05). "
           "Increase trials or check score variance.")
    )

    print(f"[Stats] {interpretation}")

    return {
        "statistic":      round(statistic, 4),
        "p_value":        round(p_value, 6),
        "significant":    significant,
        "n_pipelines":    len(pipelines),
        "n_blocks":       int(valid_cols.sum()),
        "interpretation": interpretation
    }


# ------------------------------------------------------------------ #
#  WILCOXON POST-HOC                                                   #
# ------------------------------------------------------------------ #

def wilcoxon_posthoc(shs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank tests between all pipeline pairs.
    Run this after Friedman test confirms significance.
    Identifies WHICH pairs of pipelines differ significantly.

    Applies Bonferroni correction for multiple comparisons.

    Returns
    -------
    DataFrame with columns:
        pipeline_A, pipeline_B, statistic, p_value,
        p_corrected, significant, effect_size (r)
    """
    pipelines = shs_df["pipeline_id"].unique()
    pairs     = list(combinations(pipelines, 2))
    n_pairs   = len(pairs)
    rows      = []

    for pid_a, pid_b in pairs:
        scores_a = shs_df[shs_df["pipeline_id"] == pid_a]["SHS"].values
        scores_b = shs_df[shs_df["pipeline_id"] == pid_b]["SHS"].values

        # Match lengths
        min_len  = min(len(scores_a), len(scores_b))
        scores_a = scores_a[:min_len]
        scores_b = scores_b[:min_len]

        if min_len < 10:
            print(f"[Stats] Warning: {pid_a} vs {pid_b} has only "
                  f"{min_len} paired observations. Results may be unreliable.")

        try:
            stat, p_val = stats.wilcoxon(scores_a, scores_b)
            # Bonferroni correction
            p_corrected = min(p_val * n_pairs, 1.0)
            # Effect size r = Z / sqrt(N)
            z_score     = stats.norm.ppf(1 - p_val / 2)
            effect_size = abs(z_score) / np.sqrt(min_len)

            rows.append({
                "pipeline_A":   pid_a,
                "pipeline_B":   pid_b,
                "statistic":    round(stat, 4),
                "p_value":      round(p_val, 6),
                "p_corrected":  round(p_corrected, 6),
                "significant":  p_corrected < 0.05,
                "effect_size_r":round(effect_size, 4),
                "effect_label": _effect_label(effect_size),
                "n_pairs":      min_len
            })

        except Exception as e:
            print(f"[Stats] Wilcoxon failed for {pid_a} vs {pid_b}: {e}")

    result_df = pd.DataFrame(rows)
    if len(result_df) > 0:
        print("\n[Stats] Wilcoxon Post-hoc Results (Bonferroni corrected):")
        print(result_df[["pipeline_A", "pipeline_B",
                          "p_corrected", "significant",
                          "effect_label"]].to_string(index=False))
    return result_df


def _effect_label(r: float) -> str:
    """Cohen's effect size interpretation for r."""
    if r < 0.1:   return "negligible"
    elif r < 0.3: return "small"
    elif r < 0.5: return "medium"
    else:         return "large"


# ------------------------------------------------------------------ #
#  SPEARMAN CORRELATION — SEVERITY VS SHS                             #
# ------------------------------------------------------------------ #

def spearman_severity_shs(shs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Spearman correlation between severity level and SHS score
    for each pipeline separately.

    Expected: negative correlation (higher severity → lower SHS)
    If Spearman rho is strongly negative and significant, it validates
    that your fault injection is meaningful — harder faults genuinely
    stress pipelines more.

    Returns
    -------
    DataFrame with columns:
        pipeline_id, spearman_rho, p_value, significant, interpretation
    """
    rows = []
    for pid in shs_df["pipeline_id"].unique():
        subset   = shs_df[shs_df["pipeline_id"] == pid]
        severity = subset["severity"].values
        shs      = subset["SHS"].values

        rho, p_val = stats.spearmanr(severity, shs)
        rows.append({
            "pipeline_id":    pid,
            "spearman_rho":   round(rho, 4),
            "p_value":        round(p_val, 6),
            "significant":    p_val < 0.05,
            "interpretation": _spearman_label(rho, p_val)
        })

    result_df = pd.DataFrame(rows)
    print("\n[Stats] Spearman Severity-SHS Correlation:")
    print(result_df[["pipeline_id", "spearman_rho",
                      "p_value", "interpretation"]].to_string(index=False))
    return result_df


def _spearman_label(rho: float, p: float) -> str:
    if p >= 0.05:
        return "not significant"
    if rho < -0.5:
        return "strong negative — severity meaningfully degrades SHS"
    elif rho < -0.2:
        return "moderate negative — severity partially degrades SHS"
    elif rho < 0.2:
        return "near zero — SHS insensitive to severity"
    else:
        return "positive — unexpected, investigate"


# ------------------------------------------------------------------ #
#  CONFIDENCE INTERVALS                                                #
# ------------------------------------------------------------------ #

def compute_ci95(values: np.ndarray) -> tuple[float, float]:
    """
    Computes 95% confidence interval for a set of values.
    Returns (lower_bound, upper_bound).
    """
    n    = len(values)
    mean = np.mean(values)
    se   = stats.sem(values)
    ci   = se * stats.t.ppf(0.975, df=n - 1)
    return round(mean - ci, 4), round(mean + ci, 4)


def pipeline_summary_table(shs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates the main results table for the paper.
    One row per pipeline with mean ± CI for each SHS dimension.

    This is Table 1 in your paper.
    """
    rows = []
    for pid in sorted(shs_df["pipeline_id"].unique()):
        subset = shs_df[shs_df["pipeline_id"] == pid]
        row    = {"Pipeline": pid}

        for dim in ["D", "R", "C", "S", "SHS"]:
            vals        = subset[dim].values
            mean        = np.mean(vals)
            lo, hi      = compute_ci95(vals)
            ci          = (hi - lo) / 2
            row[f"{dim}_mean"] = round(mean, 3)
            row[f"{dim}_ci"]   = round(ci, 3)
            row[f"{dim}_str"]  = f"{mean:.3f} ± {ci:.3f}"

        row["heal_rate"] = f"{subset['healed'].mean():.1%}"
        row["n_trials"]  = len(subset)
        rows.append(row)

    result_df = pd.DataFrame(rows)
    print("\n[Stats] Pipeline Summary Table (Table 1):")
    display_cols = ["Pipeline", "D_str", "R_str",
                    "C_str", "S_str", "SHS_str", "heal_rate"]
    print(result_df[display_cols].to_string(index=False))
    return result_df


# ------------------------------------------------------------------ #
#  FULL STATISTICAL REPORT                                             #
# ------------------------------------------------------------------ #

def run_full_analysis(shs_csv_path: str,
                      weights: tuple = (0.25, 0.30, 0.25, 0.20)) -> dict:
    """
    Master function. Call this once after all experiments are done.
    Runs all statistical tests and returns a complete report dict.

    Usage:
        report = run_full_analysis("results/shs_results.csv")

    Returns dict with all test results — feed into plotting functions.
    """
    print("=" * 55)
    print("SH-Bench Full Statistical Analysis")
    print("=" * 55)

    shs_df = pd.read_csv(shs_csv_path)
    print(f"\nLoaded {len(shs_df)} rows from {shs_csv_path}")
    print(f"Pipelines: {sorted(shs_df['pipeline_id'].unique())}")
    print(f"Fault types: {sorted(shs_df['fault_type'].unique())}")
    print(f"Severities: {sorted(shs_df['severity'].unique())}")

    report = {}

    # 1. Summary table
    print("\n--- Summary Table ---")
    report["summary_table"] = pipeline_summary_table(shs_df)

    # 2. Friedman test
    print("\n--- Friedman Test ---")
    report["friedman"] = friedman_test(shs_df)

    # 3. Wilcoxon post-hoc (only if Friedman significant)
    if report["friedman"]["significant"]:
        print("\n--- Wilcoxon Post-hoc ---")
        report["wilcoxon"] = wilcoxon_posthoc(shs_df)
    else:
        print("\n[Stats] Skipping Wilcoxon — Friedman not significant.")
        report["wilcoxon"] = None

    # 4. Spearman severity correlation
    print("\n--- Spearman Severity-SHS Correlation ---")
    report["spearman"] = spearman_severity_shs(shs_df)

    print("\n" + "=" * 55)
    print("Analysis complete. Feed report into plotter.py")
    print("=" * 55)

    return report


# ------------------------------------------------------------------ #
#  SELF TEST
#  python evaluator/stats.py
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import os
    print("=" * 55)
    print("Stats Module Self-Test")
    print("=" * 55)

    # Generate realistic dummy SHS data
    rng = np.random.default_rng(42)
    pipelines  = ["P1", "P2", "P3", "P4"]
    faults     = ["statistical_drift", "endpoint_kill",
                  "label_poison", "batch_corruption"]
    severities = [0.1, 0.3, 0.5]
    n_trials   = 10

    rows = []
    # Simulate P1 best at drift, P2 best at infra, realistic spread
    base_shs = {"P1": 0.72, "P2": 0.65, "P3": 0.58, "P4": 0.61}

    for pid in pipelines:
        for fault in faults:
            for sev in severities:
                for trial in range(n_trials):
                    shs = base_shs[pid] - (sev * 0.2) + rng.normal(0, 0.05)
                    shs = np.clip(shs, 0, 1)
                    rows.append({
                        "pipeline_id": pid,
                        "fault_type":  fault,
                        "severity":    sev,
                        "trial":       trial,
                        "D": np.clip(shs + rng.normal(0, 0.03), 0, 1),
                        "R": np.clip(shs + rng.normal(0, 0.03), 0, 1),
                        "C": np.clip(shs + rng.normal(0, 0.02), 0, 1),
                        "S": np.clip(shs + rng.normal(0, 0.04), 0, 1),
                        "SHS":    shs,
                        "healed": shs > 0.55,
                        "ttd":    rng.uniform(1, 60),
                        "ttr":    rng.uniform(5, 120),
                    })

    dummy_df = pd.DataFrame(rows)

    # Save to temp CSV and run full analysis
    import tempfile
    tmp = tempfile.mktemp(suffix=".csv")
    dummy_df.to_csv(tmp, index=False)

    report = run_full_analysis(tmp)

    assert "summary_table" in report
    assert "friedman"      in report
    assert "spearman"      in report

    print("\n✅ All stats tests passed. Ready for real experiment data.")
    os.unlink(tmp)