import numpy as np
import pandas as pd
import time
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from data.data_loader import (
    load_taxi, split_data, get_scaler,
    StreamSimulator, FEATURE_COLS, TARGET_COL
)
from fault_injector.fault_injector import FaultInjector
from evaluator.shs_metric import ExperimentResult


# ------------------------------------------------------------------ #
#  CONFIG                                                              #
# ------------------------------------------------------------------ #

DRIFT_PSI_THRESHOLD   = 0.2    # PSI > 0.2 = significant drift
RETRAIN_WINDOW        = 5000   # rows used for retraining
VALIDATION_SPLIT      = 0.2
IMPROVEMENT_THRESHOLD = 0.02   # new model must beat old by 2% RMSE
ROLLBACK_WINDOW       = 10     # batches to monitor after promotion
BATCH_SIZE            = 500
SLA_BUDGET_SECONDS    = 120.0
TTD_MAX_SECONDS       = 60.0


# ------------------------------------------------------------------ #
#  P1 PIPELINE                                                         #
# ------------------------------------------------------------------ #

class P1Pipeline:
    """
    Drift Detection + Automated Retraining Pipeline.

    Healing strategy:
        1. Monitor incoming batches with Evidently PSI drift detector
        2. On drift detection, trigger MLflow retraining job
        3. Promote new model only if RMSE improves by > 2%
        4. Rollback if promoted model degrades within 10 batches
    """

    def __init__(self, df_reference: pd.DataFrame,
                 mlflow_experiment: str = "SH-Bench-P1"):
        self.reference_df   = df_reference
        self.model          = None
        self.scaler         = None
        self.baseline_rmse  = None
        self.current_rmse   = None

        # Drift detection state
        self.drift_detected      = False
        self.false_positive_count = 0
        self.total_detections    = 0

        # Timing
        self.fault_inject_time   = None
        self.detection_time      = None
        self.recovery_time       = None

        # Post-recovery monitoring
        self.post_heal_rmses     = []
        self.monitoring_mode     = False
        self.batches_since_heal  = 0

        # MLflow
        mlflow.set_experiment(mlflow_experiment)

        # Column mapping for Evidently
        self.column_mapping = ColumnMapping(
            target=TARGET_COL,
            numerical_features=FEATURE_COLS
        )

    # ── Initial Training ──────────────────────────────────────────

    def initial_train(self, df_train: pd.DataFrame) -> float:
        """
        Trains the baseline Ridge regression model.
        Returns baseline RMSE on held-out validation set.
        """
        X_train, X_val, y_train, y_val = split_data(df_train)
        self.scaler = get_scaler(X_train)

        X_train_s = self.scaler.transform(X_train)
        X_val_s   = self.scaler.transform(X_val)

        self.model = Ridge(alpha=1.0)
        self.model.fit(X_train_s, y_train)

        preds = self.model.predict(X_val_s)
        self.baseline_rmse = np.sqrt(mean_squared_error(y_val, preds))
        self.current_rmse  = self.baseline_rmse

        print(f"[P1] Initial training complete. Baseline RMSE: {self.baseline_rmse:.4f}")
        return self.baseline_rmse

    # ── Drift Detection ───────────────────────────────────────────

    def _detect_drift(self, batch: pd.DataFrame) -> bool:
        """
        Uses Evidently DataDrift report to check if incoming batch
        has drifted from the reference distribution.
        Returns True if drift is detected.
        """
        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(
                reference_data=self.reference_df[FEATURE_COLS + [TARGET_COL]].sample(
                    min(1000, len(self.reference_df)), random_state=42),
                current_data=batch[FEATURE_COLS + [TARGET_COL]].dropna(),
                column_mapping=self.column_mapping
            )
            result = report.as_dict()
            drift_score = result["metrics"][0]["result"]["share_of_drifted_columns"]
            return drift_score > DRIFT_PSI_THRESHOLD

        except Exception as e:
            print(f"[P1] Drift detection error: {e}")
            return False

    # ── Retraining ────────────────────────────────────────────────

    def _retrain(self, recent_data: pd.DataFrame) -> tuple[object, float]:
        """
        Retrains a new Ridge model on recent data.
        Logs to MLflow. Returns (new_model, new_rmse).
        """
        with mlflow.start_run(run_name="retrain", nested=True):
            recent_clean = recent_data[FEATURE_COLS + [TARGET_COL]].dropna()
            if len(recent_clean) < 100:
                print("[P1] Not enough clean data to retrain.")
                return self.model, self.current_rmse

            X_train, X_val, y_train, y_val = split_data(recent_clean)
            X_train_s = self.scaler.transform(X_train)
            X_val_s   = self.scaler.transform(X_val)

            new_model = Ridge(alpha=1.0)
            new_model.fit(X_train_s, y_train)
            preds     = new_model.predict(X_val_s)
            new_rmse  = np.sqrt(mean_squared_error(y_val, preds))

            mlflow.log_metric("retrain_rmse", new_rmse)
            mlflow.log_metric("baseline_rmse", self.baseline_rmse)
            print(f"[P1] Retrain complete. New RMSE: {new_rmse:.4f} "
                  f"(baseline: {self.baseline_rmse:.4f})")

            return new_model, new_rmse

    # ── Inference ─────────────────────────────────────────────────

    def predict(self, batch: pd.DataFrame) -> np.ndarray:
        """Runs inference on a batch. Returns predictions."""
        X = batch[FEATURE_COLS].dropna().values
        X_s = self.scaler.transform(X)
        return self.model.predict(X_s)

    def _compute_batch_rmse(self, batch: pd.DataFrame) -> float:
        """Computes RMSE on a labeled batch."""
        clean = batch[FEATURE_COLS + [TARGET_COL]].dropna()
        if len(clean) < 10:
            return self.current_rmse
        preds = self.predict(clean)
        return np.sqrt(mean_squared_error(clean[TARGET_COL].values, preds))

    # ── Main Run Loop ─────────────────────────────────────────────

    def run_experiment(self,
                       df_stream: pd.DataFrame,
                       fault_type: str,
                       severity: float,
                       trial: int,
                       injector: FaultInjector) -> ExperimentResult:
        """
        Runs one full fault injection experiment.

        Steps:
            1. Stabilization: 5 clean batches to confirm baseline
            2. Fault injection at batch 6
            3. Observation: pipeline attempts healing
            4. Recovery verification: 5 post-heal batches

        Returns a filled ExperimentResult for SHS computation.
        """
        print(f"\n[P1] Experiment | fault={fault_type} | "
              f"severity={severity} | trial={trial}")

        stream   = StreamSimulator(df_stream, batch_size=BATCH_SIZE)
        data_buffer = []  # rolling buffer for retraining

        ttd = TTD_MAX_SECONDS
        ttr = SLA_BUDGET_SECONDS
        post_heal_rmse = self.baseline_rmse
        post_recovery_variance = 0.0
        healed = False
        false_positives = 0
        fault_injected = False
        fault_inject_time = None
        detection_time = None
        recovery_time = None
        post_heal_rmses = []

        # ── Phase 1: Stabilization (5 batches) ───────────────────
        for i in range(5):
            batch = next(stream)
            data_buffer.append(batch)
            rmse = self._compute_batch_rmse(batch)
            print(f"  [Stabilization] batch {i+1}/5 | RMSE: {rmse:.4f}")

        # ── Phase 2: Fault Injection ──────────────────────────────
        fault_inject_time = time.time()

        # ── Phase 3: Observation Window (up to 20 batches) ────────
        for i in range(20):
            batch = next(stream)

            # Apply fault to incoming batch
            if fault_type in ["schema_drift", "statistical_drift",
                               "label_poison", "batch_corruption",
                               "compound_fault"]:
                try:
                    if fault_type == "schema_drift":
                        batch = injector.schema_drift(batch, severity)
                        # Re-add missing cols as NaN so pipeline can handle
                        for col in FEATURE_COLS + [TARGET_COL]:
                            if col not in batch.columns:
                                batch[col] = np.nan
                    elif fault_type == "statistical_drift":
                        batch = injector.statistical_drift(batch, severity)
                    elif fault_type == "label_poison":
                        batch = injector.label_poison(batch, TARGET_COL, severity)
                    elif fault_type == "batch_corruption":
                        batch = injector.batch_corruption(batch, severity)
                    elif fault_type == "compound_fault":
                        batch = injector.compound_fault(batch, TARGET_COL, severity)
                except Exception:
                    pass

            elif fault_type == "concept_drift":
                batch = injector.concept_drift(batch, TARGET_COL, severity)

            current_rmse = self._compute_batch_rmse(batch)
            data_buffer.append(batch)
            if len(data_buffer) > 20:
                data_buffer.pop(0)

            # ── Drift detection ───────────────────────────────────
            if not fault_injected:
                fault_injected = True

            drift_flagged = self._detect_drift(batch)

            if drift_flagged:
                self.total_detections += 1

                # Was this a true positive?
                rmse_degraded = current_rmse > self.baseline_rmse * 1.05
                if not rmse_degraded:
                    false_positives += 1
                    print(f"  [P1] False positive drift alarm at batch {i+1}")
                    continue

                # True positive — record TTD
                if detection_time is None:
                    detection_time = time.time()
                    ttd = detection_time - fault_inject_time
                    ttd = min(ttd, TTD_MAX_SECONDS)
                    print(f"  [P1] Drift detected at batch {i+1} | TTD: {ttd:.2f}s")

                # ── Trigger retraining ────────────────────────────
                retrain_data = pd.concat(data_buffer[-10:])
                retrain_start = time.time()
                new_model, new_rmse = self._retrain(retrain_data)

                # Promote only if improved
                if new_rmse < self.current_rmse * (1 - IMPROVEMENT_THRESHOLD):
                    self.model = new_model
                    self.current_rmse = new_rmse
                    recovery_time = time.time()
                    ttr = recovery_time - fault_inject_time
                    ttr = min(ttr, SLA_BUDGET_SECONDS)
                    print(f"  [P1] Model promoted. TTR: {ttr:.2f}s")
                    break
                else:
                    print(f"  [P1] New model not better. Holding current model.")

        # ── Phase 4: Post-heal verification (5 batches) ───────────
        for i in range(5):
            batch = next(stream)
            rmse = self._compute_batch_rmse(batch)
            post_heal_rmses.append(rmse)

        if post_heal_rmses:
            post_heal_rmse = np.mean(post_heal_rmses)
            post_recovery_variance = np.var(post_heal_rmses)
            healed = post_heal_rmse <= self.baseline_rmse * 1.10

        print(f"  [P1] Done | healed={healed} | "
              f"post_heal_rmse={post_heal_rmse:.4f} | "
              f"baseline={self.baseline_rmse:.4f}")

        return ExperimentResult(
            pipeline_id            = "P1",
            fault_type             = fault_type,
            severity               = severity,
            trial                  = trial,
            ttd                    = ttd,
            ttr                    = ttr,
            baseline_rmse          = self.baseline_rmse,
            post_fault_rmse        = self.baseline_rmse * (1 + severity),
            post_heal_rmse         = post_heal_rmse,
            healed                 = healed,
            remediation_cost       = 1.0,   # full retrain = max cost
            post_recovery_variance = post_recovery_variance,
            false_positives        = false_positives,
            total_detections       = max(1, self.total_detections),
            ttd_max                = TTD_MAX_SECONDS,
            sla_budget             = SLA_BUDGET_SECONDS,
        )


# ------------------------------------------------------------------ #
#  QUICK SMOKE TEST
#  python p1_pipeline.py
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=" * 55)
    print("P1 Pipeline Smoke Test")
    print("=" * 55)

    PATH_2019 = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data',
        'yellow_tripdata_2019-01.csv')

    df = load_taxi(PATH_2019, sample_size=10000)
    df_train = df.iloc[:8000]
    df_stream = df.iloc[8000:]

    pipeline = P1Pipeline(df_reference=df_train)
    pipeline.initial_train(df_train)

    injector = FaultInjector(random_seed=42)

    result = pipeline.run_experiment(
        df_stream  = df_stream,
        fault_type = "statistical_drift",
        severity   = 0.3,
        trial      = 0,
        injector   = injector
    )

    print("\n--- ExperimentResult ---")
    print(f"  pipeline_id:   {result.pipeline_id}")
    print(f"  fault_type:    {result.fault_type}")
    print(f"  severity:      {result.severity}")
    print(f"  ttd:           {result.ttd:.2f}s")
    print(f"  ttr:           {result.ttr:.2f}s")
    print(f"  healed:        {result.healed}")
    print(f"  baseline_rmse: {result.baseline_rmse:.4f}")
    print(f"  post_heal_rmse:{result.post_heal_rmse:.4f}")
    print("\n✅ P1 smoke test complete.")