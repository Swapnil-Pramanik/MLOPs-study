import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import os

# ------------------------------------------------------------------ #
#  STYLE CONFIG                                                        #
# ------------------------------------------------------------------ #

PIPELINE_COLORS = {
    "P1": "#2196F3",   # blue  — drift retrain
    "P2": "#4CAF50",   # green — circuit breaker
    "P3": "#FF9800",   # orange — RL autoscaler
    "P4": "#9C27B0",   # purple — causal RCA
}

PIPELINE_LABELS = {
    "P1": "P1: Drift+Retrain",
    "P2": "P2: Circuit Breaker",
    "P3": "P3: RL Autoscaler",
    "P4": "P4: Causal RCA",
}

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi":     150,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
})


def _ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ------------------------------------------------------------------ #
#  FIGURE 1 — Radar Chart: Per-Pipeline SHS Dimension Profile         #
# ------------------------------------------------------------------ #

def plot_radar(shs_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Radar/spider chart showing D, R, C, S scores per pipeline.
    Reveals each pipeline's healing profile at a glance.
    This is the most visually distinctive figure in the paper.
    """
    _ensure_figures_dir()
    dimensions = ["D", "R", "C", "S"]
    n_dims     = len(dimensions)
    angles     = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles    += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(6, 6),
                           subplot_kw=dict(polar=True))

    for pid in sorted(shs_df["pipeline_id"].unique()):
        subset = shs_df[shs_df["pipeline_id"] == pid]
        values = [subset[d].mean() for d in dimensions]
        values += values[:1]

        ax.plot(angles, values,
                color=PIPELINE_COLORS.get(pid, "gray"),
                linewidth=2, label=PIPELINE_LABELS.get(pid, pid))
        ax.fill(angles, values,
                color=PIPELINE_COLORS.get(pid, "gray"), alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Detection\n(D)", "Response\n(R)",
                         "Coverage\n(C)", "Stability\n(S)"],
                        fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title("Figure 1: SHS Dimension Profile per Pipeline",
                 pad=20, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "fig1_radar.pdf")
        plt.savefig(path)
        print(f"[Plotter] Saved {path}")
    return fig


# ------------------------------------------------------------------ #
#  FIGURE 2 — Box Plot: SHS Distribution Per Pipeline                 #
# ------------------------------------------------------------------ #

def plot_shs_boxplot(shs_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Box plot of SHS score distribution across all faults and trials.
    Shows median, IQR, and outliers per pipeline.
    """
    _ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(8, 5))

    pipelines = sorted(shs_df["pipeline_id"].unique())
    data      = [shs_df[shs_df["pipeline_id"] == p]["SHS"].values
                 for p in pipelines]
    colors    = [PIPELINE_COLORS.get(p, "gray") for p in pipelines]

    bp = ax.boxplot(data, patch_artist=True, notch=True,
                    medianprops=dict(color="black", linewidth=2))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels([PIPELINE_LABELS.get(p, p) for p in pipelines],
                        rotation=15, ha="right")
    ax.set_ylabel("SHS Score")
    ax.set_title("Figure 2: SHS Score Distribution per Pipeline",
                 fontweight="bold")
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color="gray", linestyle="--",
               alpha=0.5, label="Baseline threshold (0.5)")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "fig2_boxplot.pdf")
        plt.savefig(path)
        print(f"[Plotter] Saved {path}")
    return fig


# ------------------------------------------------------------------ #
#  FIGURE 3 — Heatmap: SHS per Pipeline × Fault Class                 #
# ------------------------------------------------------------------ #

def plot_heatmap(shs_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Heatmap of mean SHS score for each (pipeline, fault_type) combination.
    Immediately shows which pipelines struggle with which fault types.
    This is the most information-dense figure in the paper.
    """
    _ensure_figures_dir()
    pivot = shs_df.groupby(["pipeline_id", "fault_type"])["SHS"].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot,
                annot=True, fmt=".2f",
                cmap="RdYlGn",
                vmin=0, vmax=1,
                linewidths=0.5,
                linecolor="white",
                cbar_kws={"label": "Mean SHS Score"},
                ax=ax)

    ax.set_title("Figure 3: SHS Score Heatmap (Pipeline × Fault Type)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Fault Type")
    ax.set_ylabel("Pipeline")
    ax.set_yticklabels([PIPELINE_LABELS.get(p, p)
                        for p in pivot.index], rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "fig3_heatmap.pdf")
        plt.savefig(path)
        print(f"[Plotter] Saved {path}")
    return fig


# ------------------------------------------------------------------ #
#  FIGURE 4 — Line Plot: SHS vs Severity Level Per Pipeline           #
# ------------------------------------------------------------------ #

def plot_severity_curve(shs_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Line plot of mean SHS ± 95% CI vs severity level for each pipeline.
    Shows how each pipeline degrades under increasing fault intensity.
    The slope of each line reflects fault tolerance.
    """
    _ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(8, 5))

    for pid in sorted(shs_df["pipeline_id"].unique()):
        subset     = shs_df[shs_df["pipeline_id"] == pid]
        sev_groups = subset.groupby("severity")["SHS"]

        severities = sorted(subset["severity"].unique())
        means = [sev_groups.get_group(s).mean() for s in severities]
        cis   = [stats.sem(sev_groups.get_group(s).values) *
                 stats.t.ppf(0.975, df=len(sev_groups.get_group(s)) - 1)
                 for s in severities]

        color = PIPELINE_COLORS.get(pid, "gray")
        ax.plot(severities, means,
                marker="o", linewidth=2,
                color=color,
                label=PIPELINE_LABELS.get(pid, pid))
        ax.fill_between(severities,
                         [m - c for m, c in zip(means, cis)],
                         [m + c for m, c in zip(means, cis)],
                         color=color, alpha=0.15)

    ax.set_xlabel("Fault Severity (σ)")
    ax.set_ylabel("Mean SHS Score ± 95% CI")
    ax.set_title("Figure 4: SHS Degradation Under Increasing Fault Severity",
                 fontweight="bold")
    ax.set_xticks([0.1, 0.3, 0.5])
    ax.set_xticklabels(["Low (0.1)", "Medium (0.3)", "High (0.5)"])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "fig4_severity_curve.pdf")
        plt.savefig(path)
        print(f"[Plotter] Saved {path}")
    return fig


# ------------------------------------------------------------------ #
#  FIGURE 5 — Bar Chart: Fault Coverage (C Dimension) Per Pipeline    #
# ------------------------------------------------------------------ #

def plot_fault_coverage(shs_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Grouped bar chart of fault coverage (heal rate) per pipeline
    broken down by fault type.
    Shows which pipeline can handle which fault class.
    """
    _ensure_figures_dir()
    heal_rates = shs_df.groupby(
        ["pipeline_id", "fault_type"])["healed"].mean().reset_index()
    heal_rates.columns = ["pipeline_id", "fault_type", "heal_rate"]

    fault_types = sorted(heal_rates["fault_type"].unique())
    pipelines   = sorted(heal_rates["pipeline_id"].unique())
    n_faults    = len(fault_types)
    n_pipes     = len(pipelines)
    x           = np.arange(n_faults)
    width       = 0.8 / n_pipes

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, pid in enumerate(pipelines):
        subset = heal_rates[heal_rates["pipeline_id"] == pid]
        rates  = [subset[subset["fault_type"] == f]["heal_rate"].values[0]
                  if f in subset["fault_type"].values else 0
                  for f in fault_types]

        bars = ax.bar(x + i * width - (n_pipes - 1) * width / 2,
                      rates, width * 0.9,
                      label=PIPELINE_LABELS.get(pid, pid),
                      color=PIPELINE_COLORS.get(pid, "gray"),
                      alpha=0.8)

        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            if rate > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{rate:.0%}",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in fault_types],
                        fontsize=9)
    ax.set_ylabel("Heal Rate (Fault Coverage)")
    ax.set_ylim(0, 1.15)
    ax.set_title("Figure 5: Fault Coverage per Pipeline and Fault Type",
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "fig5_fault_coverage.pdf")
        plt.savefig(path)
        print(f"[Plotter] Saved {path}")
    return fig


# ------------------------------------------------------------------ #
#  FIGURE 6 — Sensitivity Analysis: SHS Rank Stability                #
# ------------------------------------------------------------------ #

def plot_sensitivity(sensitivity_df: pd.DataFrame,
                     save: bool = True) -> plt.Figure:
    """
    Visualizes SHS rank stability under 100 random weight perturbations.
    Each line = one pipeline's SHS score across all perturbations.
    Tightly clustered lines = stable rankings = robust metric.

    Parameters
    ----------
    sensitivity_df : output of sensitivity_analysis() from shs_metric.py
    """
    _ensure_figures_dir()
    shs_cols   = [c for c in sensitivity_df.columns if c.startswith("SHS_")]
    pipelines  = [c.replace("SHS_", "") for c in shs_cols]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: SHS scores across perturbations
    for col, pid in zip(shs_cols, pipelines):
        scores = sensitivity_df[col].values
        ax1.plot(scores, alpha=0.6,
                 color=PIPELINE_COLORS.get(pid, "gray"),
                 label=PIPELINE_LABELS.get(pid, pid),
                 linewidth=1)

    ax1.set_xlabel("Weight Perturbation Index")
    ax1.set_ylabel("SHS Score")
    ax1.set_title("SHS Scores Across 100\nRandom Weight Vectors",
                  fontweight="bold")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=8)
    ax1.yaxis.grid(True, alpha=0.3)

    # Right: Rank distribution as box plot
    rank_cols = [c for c in sensitivity_df.columns if c.startswith("rank_")]
    rank_pids = [c.replace("rank_", "") for c in rank_cols]
    rank_data = [sensitivity_df[c].values for c in rank_cols]
    colors    = [PIPELINE_COLORS.get(p, "gray") for p in rank_pids]

    bp = ax2.boxplot(rank_data, patch_artist=True,
                     medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_xticklabels([PIPELINE_LABELS.get(p, p)
                          for p in rank_pids],
                          rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("Rank (1 = best)")
    ax2.set_title("Rank Distribution Across\n100 Weight Perturbations",
                  fontweight="bold")
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # rank 1 at top

    plt.suptitle("Figure 6: SHS Metric Sensitivity Analysis",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "fig6_sensitivity.pdf")
        plt.savefig(path)
        print(f"[Plotter] Saved {path}")
    return fig


# ------------------------------------------------------------------ #
#  MASTER PLOT FUNCTION                                                #
# ------------------------------------------------------------------ #

def generate_all_figures(shs_csv: str,
                          sensitivity_df: pd.DataFrame = None):
    """
    Generates all 6 paper figures from shs_results.csv.

    Usage on Day 3:
        from evaluator.plotter import generate_all_figures
        from evaluator.shs_metric import sensitivity_analysis
        import pandas as pd

        # Load raw results and reconstruct ExperimentResult objects
        # then run:
        generate_all_figures("results/shs_results.csv", sens_df)
    """
    shs_df = pd.read_csv(shs_csv)

    print("[Plotter] Generating all 6 figures...")
    plot_radar(shs_df)
    plot_shs_boxplot(shs_df)
    plot_heatmap(shs_df)
    plot_severity_curve(shs_df)
    plot_fault_coverage(shs_df)

    if sensitivity_df is not None:
        plot_sensitivity(sensitivity_df)
    else:
        print("[Plotter] Skipping Fig 6 — no sensitivity_df provided.")

    print(f"\n[Plotter] All figures saved to {FIGURES_DIR}")


# ------------------------------------------------------------------ #
#  SELF TEST
#  python evaluator/plotter.py
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=" * 55)
    print("Plotter Self-Test")
    print("=" * 55)

    rng = np.random.default_rng(42)
    pipelines  = ["P1", "P2", "P3", "P4"]
    faults     = ["statistical_drift", "endpoint_kill",
                  "label_poison", "batch_corruption",
                  "concept_drift", "compound_fault"]
    severities = [0.1, 0.3, 0.5]

    base_shs = {"P1": 0.72, "P2": 0.65, "P3": 0.58, "P4": 0.61}
    rows = []
    for pid in pipelines:
        for fault in faults:
            for sev in severities:
                for trial in range(10):
                    shs = base_shs[pid] - sev * 0.2 + rng.normal(0, 0.06)
                    shs = float(np.clip(shs, 0, 1))
                    rows.append({
                        "pipeline_id": pid,
                        "fault_type":  fault,
                        "severity":    sev,
                        "trial":       trial,
                        "D":    float(np.clip(shs + rng.normal(0, 0.04), 0, 1)),
                        "R":    float(np.clip(shs + rng.normal(0, 0.04), 0, 1)),
                        "C":    float(np.clip(shs + rng.normal(0, 0.03), 0, 1)),
                        "S":    float(np.clip(shs + rng.normal(0, 0.05), 0, 1)),
                        "SHS":  shs,
                        "healed": shs > 0.5,
                        "ttd":  float(rng.uniform(1, 60)),
                        "ttr":  float(rng.uniform(5, 120)),
                    })

    dummy_df = pd.DataFrame(rows)

    import tempfile
    tmp_csv = tempfile.mktemp(suffix=".csv")
    dummy_df.to_csv(tmp_csv, index=False)

    # Test all plots
    plot_radar(dummy_df, save=True)
    print("✅ Figure 1 (radar) generated")

    plot_shs_boxplot(dummy_df, save=True)
    print("✅ Figure 2 (boxplot) generated")

    plot_heatmap(dummy_df, save=True)
    print("✅ Figure 3 (heatmap) generated")

    plot_severity_curve(dummy_df, save=True)
    print("✅ Figure 4 (severity curve) generated")

    plot_fault_coverage(dummy_df, save=True)
    print("✅ Figure 5 (fault coverage) generated")

    # Sensitivity plot needs sensitivity_df
    from evaluator.shs_metric import ExperimentResult, sensitivity_analysis
    dummy_results = []
    for _, row in dummy_df.iterrows():
        dummy_results.append(ExperimentResult(
            pipeline_id=row["pipeline_id"],
            fault_type=row["fault_type"],
            severity=row["severity"],
            trial=int(row["trial"]),
            ttd=row["ttd"], ttr=row["ttr"],
            baseline_rmse=4.5,
            post_fault_rmse=5.5,
            post_heal_rmse=4.5 + rng.uniform(0, 0.5),
            healed=bool(row["healed"]),
            remediation_cost=float(rng.uniform(0.1, 1.0)),
            post_recovery_variance=float(rng.uniform(0.01, 0.3)),
        ))

    sens_df = sensitivity_analysis(dummy_results, n_perturbations=100)
    plot_sensitivity(sens_df, save=True)
    print("✅ Figure 6 (sensitivity) generated")

    print(f"\n✅ All 6 figures generated and saved to paper/figures/")
    os.unlink(tmp_csv)