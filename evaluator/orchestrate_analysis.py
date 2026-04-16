import pandas as pd
import numpy as np
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluator.shs_metric import ExperimentResult, compute_pipeline_SHS
from evaluator.stats import (
    pipeline_summary_table, friedman_test, 
    wilcoxon_posthoc, spearman_severity_shs
)
from evaluator import plotter

# 1. SETUP
RAW_PATH = "results/raw_results.csv"
SHS_PATH = "results/shs_results.csv"
PLOTS_DIR = "results/plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

# Monkey-patch plotter to save to results/plots/
plotter.FIGURES_DIR = PLOTS_DIR

def run_analysis():
    print("=" * 60)
    print("SH-Bench Master Orchestration Script")
    print("=" * 60)

    # 2. LOAD DATA
    print(f"\n[1/3] Loading data from {RAW_PATH}...")
    try:
        raw_df = pd.read_csv(RAW_PATH)
        shs_df_file = pd.read_csv(SHS_PATH)
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return

    # 3. SANITIZE
    raw_df = raw_df.dropna()
    shs_df_file = shs_df_file.dropna()

    # 4. PROCESS RAW DATA INTO SHS (for stats requirement)
    # The user says stats MUST use raw_results.csv and not aggregated data.
    # We reconstruct per-trial SHS from raw observations.
    print("[2/3] Reconstructing granular SHS from raw observations...")
    results = []
    for _, row in raw_df.iterrows():
        results.append(ExperimentResult(
            pipeline_id=row['pipeline_id'],
            fault_type=row['fault_type'],
            severity=row['severity'],
            trial=int(row['trial']),
            ttd=row['ttd'],
            ttr=row['ttr'],
            baseline_rmse=row['baseline_rmse'],
            post_fault_rmse=row['post_fault_rmse'],
            post_heal_rmse=row['post_heal_rmse'],
            healed=bool(row['healed']),
            remediation_cost=row['remediation_cost'],
            post_recovery_variance=row['post_recovery_variance'],
            false_positives=int(row['false_positives']),
            total_detections=int(row['total_detections'])
        ))
    
    # Group by pipeline to compute C correctly (as per shs_metric logic)
    all_computed_shs = []
    for pid in raw_df['pipeline_id'].unique():
        pid_results = [r for r in results if r.pipeline_id == pid]
        computed_df = compute_pipeline_SHS(pid_results)
        all_computed_shs.append(computed_df)
    
    shs_df = pd.concat(all_computed_shs).reset_index(drop=True)

    # 5. EXECUTE STATS (using calculated granular data)
    print("\n[3/3] Running Statistical Analysis...")
    print("-" * 40)
    
    # Summary Table
    pipeline_summary_table(shs_df)
    
    # Friedman Test
    friedman_results = friedman_test(shs_df)
    
    # Wilcoxon (only if significant)
    if friedman_results['significant']:
        wilcoxon_posthoc(shs_df)
    
    # Spearman Correlation
    spearman_severity_shs(shs_df)

    # 6. EXECUTE PLOTTING
    print("\n[4/4] Generating Plots in " + PLOTS_DIR + "...")
    
    # We use shs_df (calculated from raw) for these as per instructions 
    # (The SHS summary plots use SHS labels)
    
    # Radar (Fig 1)
    plotter.plot_radar(shs_df, save=True)
    # Box Plot (Fig 2)
    plotter.plot_shs_boxplot(shs_df, save=True)
    # Heatmap (Fig 3)
    plotter.plot_heatmap(shs_df, save=True)
    # Severity Curve (Fig 4)
    plotter.plot_severity_curve(shs_df, save=True)
    # Fault Coverage (Fig 5)
    plotter.plot_fault_coverage(shs_df, save=True)

    print("\n" + "=" * 60)
    print("Analysis Complete. All artifacts saved.")
    print("=" * 60)

if __name__ == "__main__":
    run_analysis()
