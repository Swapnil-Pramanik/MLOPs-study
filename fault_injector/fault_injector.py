import numpy as np
import pandas as pd
from copy import deepcopy


class FaultInjector:
    """
    Parameterized fault injector for SH-Bench.
    Covers 8 fault scenarios across 3 planes: Data, Model, Infrastructure.
    Each fault accepts severity sigma in {0.1, 0.3, 0.5}.
    """

    def __init__(self, random_seed=42):
        self.rng = np.random.default_rng(random_seed)
        self.active_fault = None

    # ------------------------------------------------------------------ #
    #  PLANE 1 — DATA FAULTS                                               #
    # ------------------------------------------------------------------ #

    def schema_drift(self, df: pd.DataFrame, severity: float) -> pd.DataFrame:
        """
        Drops or renames a fraction of columns proportional to severity.
        severity=0.1 → 1 column affected
        severity=0.3 → 30% of columns affected
        severity=0.5 → 50% of columns affected
        """
        df = df.copy()
        n_cols = max(1, int(len(df.columns) * severity))
        affected = self.rng.choice(df.columns.tolist(), size=n_cols, replace=False)

        for i, col in enumerate(affected):
            if i % 2 == 0:
                df = df.drop(columns=[col])           # drop every other column
            else:
                df = df.rename(columns={col: col + "_corrupted"})  # rename rest

        self.active_fault = "schema_drift"
        return df

    def statistical_drift(self, df: pd.DataFrame, severity: float,
                          numeric_only=True) -> pd.DataFrame:
        """
        Shifts numeric feature distributions by injecting Gaussian noise
        scaled by severity * column std.
        severity=0.1 → subtle shift
        severity=0.5 → severe distribution shift
        """
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in num_cols:
            std = df[col].std()
            noise = self.rng.normal(loc=severity * std * 3,
                                    scale=severity * std,
                                    size=len(df))
            df[col] = df[col] + noise

        self.active_fault = "statistical_drift"
        return df

    def label_poison(self, df: pd.DataFrame, label_col: str,
                     severity: float) -> pd.DataFrame:
        """
        Corrupts a fraction of labels equal to severity.
        For regression: adds large noise to target.
        For classification: randomly flips labels.
        severity=0.1 → 10% labels corrupted
        severity=0.5 → 50% labels corrupted
        """
        df = df.copy()
        n_corrupt = int(len(df) * severity)
        corrupt_idx = self.rng.choice(df.index, size=n_corrupt, replace=False)

        if df[label_col].dtype in [np.float32, np.float64]:
            noise = self.rng.normal(loc=0,
                                    scale=df[label_col].std() * severity * 5,
                                    size=n_corrupt)
            df.loc[corrupt_idx, label_col] += noise
        else:
            unique_labels = df[label_col].unique()
            df.loc[corrupt_idx, label_col] = self.rng.choice(unique_labels,
                                                              size=n_corrupt)

        self.active_fault = "label_poison"
        return df

    # ------------------------------------------------------------------ #
    #  PLANE 2 — MODEL FAULTS                                              #
    # ------------------------------------------------------------------ #

    def concept_drift(self, df: pd.DataFrame, label_col: str,
                      severity: float) -> pd.DataFrame:
        """
        Shifts the ground truth relationship between features and target
        by inverting or scaling the target as a function of severity.
        Simulates real world concept drift (e.g. user behaviour change).
        severity=0.1 → mild relationship weakening
        severity=0.5 → near complete target inversion
        """
        df = df.copy()
        original = df[label_col].values
        inverted = -original

        # Blend between original and inverted proportional to severity
        df[label_col] = (1 - severity) * original + severity * inverted

        self.active_fault = "concept_drift"
        return df

    def weight_corruption(self, model, severity: float):
        """
        Adds Gaussian noise to sklearn model coefficients.
        Works with LinearRegression, Ridge, LogisticRegression.
        severity=0.1 → small perturbation
        severity=0.5 → significant weight corruption
        """
        import copy
        corrupted_model = copy.deepcopy(model)

        if hasattr(corrupted_model, 'coef_'):
            noise = self.rng.normal(loc=0,
                                    scale=severity * np.abs(corrupted_model.coef_).mean(),
                                    size=corrupted_model.coef_.shape)
            corrupted_model.coef_ += noise

        if hasattr(corrupted_model, 'intercept_'):
            corrupted_model.intercept_ += self.rng.normal(
                loc=0, scale=severity * abs(float(corrupted_model.intercept_)))

        self.active_fault = "weight_corruption"
        return corrupted_model

    def endpoint_kill(self, serving_function):
        """
        Returns a wrapped function that raises a ConnectionError
        simulating a dead serving endpoint.
        severity is implicit — endpoint is fully killed.
        Usage: dead_serve = injector.endpoint_kill(original_serve_fn)
        """
        def dead_endpoint(*args, **kwargs):
            raise ConnectionError("[FAULT] Serving endpoint killed by FaultInjector.")

        self.active_fault = "endpoint_kill"
        return dead_endpoint

    # ------------------------------------------------------------------ #
    #  PLANE 3 — INFRASTRUCTURE FAULTS                                     #
    # ------------------------------------------------------------------ #

    def memory_pressure(self, severity: float):
        """
        Allocates a numpy array to simulate memory pressure.
        severity=0.1 → ~100MB allocation
        severity=0.3 → ~500MB allocation
        severity=0.5 → ~1GB allocation
        Returns the allocation handle — caller must del it to release.
        """
        mb = int(severity * 2000)  # severity 0.5 → 1000MB
        print(f"[FAULT] Allocating {mb}MB to simulate memory pressure.")
        allocation = np.ones((mb * 1024 * 1024 // 8,), dtype=np.float64)
        self.active_fault = "memory_pressure"
        return allocation  # caller: handle = injector.memory_pressure(0.3) ... del handle

    def batch_corruption(self, df: pd.DataFrame, severity: float) -> pd.DataFrame:
        """
        Randomly sets a fraction of values to NaN across the batch,
        simulating corrupted or dropped data packets in a streaming pipeline.
        severity=0.1 → 10% values NaNed
        severity=0.5 → 50% values NaNed
        """
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        n_cells = int(len(df) * len(num_cols) * severity)

        for _ in range(n_cells):
            row = self.rng.integers(0, len(df))
            col = self.rng.choice(num_cols)
            df.at[df.index[row], col] = np.nan

        self.active_fault = "batch_corruption"
        return df

    # ------------------------------------------------------------------ #
    #  COMPOUND FAULT                                                       #
    # ------------------------------------------------------------------ #

    def compound_fault(self, df: pd.DataFrame, label_col: str,
                       severity: float) -> pd.DataFrame:
        """
        Simultaneously applies statistical_drift + concept_drift.
        Tests whether RCA pipeline correctly disambiguates root cause
        when multiple fault signals are present at once.
        This is the hardest fault class in the benchmark.
        """
        df = self.statistical_drift(df, severity)
        df = self.concept_drift(df, label_col, severity)
        self.active_fault = "compound_fault"
        return df

    # ------------------------------------------------------------------ #
    #  UTILITY                                                              #
    # ------------------------------------------------------------------ #

    def get_active_fault(self):
        return self.active_fault

    def reset(self):
        self.active_fault = None


# ------------------------------------------------------------------ #
#  QUICK SELF-TEST — run this file directly to verify everything works
#  python fault_injector.py
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("Running FaultInjector self-test...\n")

    # Create dummy dataframe
    df = pd.DataFrame({
        "feature_1": np.random.normal(0, 1, 500),
        "feature_2": np.random.normal(5, 2, 500),
        "feature_3": np.random.uniform(0, 10, 500),
        "target":    np.random.normal(10, 3, 500)
    })

    injector = FaultInjector(random_seed=42)
    severities = [0.1, 0.3, 0.5]

    tests = [
        ("schema_drift",      lambda s: injector.schema_drift(df, s)),
        ("statistical_drift", lambda s: injector.statistical_drift(df, s)),
        ("label_poison",      lambda s: injector.label_poison(df, "target", s)),
        ("concept_drift",     lambda s: injector.concept_drift(df, "target", s)),
        ("batch_corruption",  lambda s: injector.batch_corruption(df, s)),
        ("compound_fault",    lambda s: injector.compound_fault(df, "target", s)),
    ]

    for fault_name, fault_fn in tests:
        for sigma in severities:
            try:
                result = fault_fn(sigma)
                print(f"  ✅  {fault_name:<22} | severity={sigma} | "
                      f"output shape: {result.shape}")
            except Exception as e:
                print(f"  ❌  {fault_name:<22} | severity={sigma} | ERROR: {e}")

    print("\nAll data/infra faults tested.")
    print("weight_corruption and endpoint_kill require a fitted model — skipped in self-test.")
    print("\nFaultInjector is ready.")