import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------ #
#  CONSTANTS                                                           #
# ------------------------------------------------------------------ #

FEATURE_COLS = [
    "trip_distance",
    "passenger_count",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "extra",
    "mta_tax",
    "tolls_amount",
    "congestion_surcharge"
]

TARGET_COL = "fare_amount"

# Reasonable fare bounds — filters out data entry errors
FARE_MIN = 2.5
FARE_MAX = 200.0
TRIP_MIN = 0.1
TRIP_MAX = 100.0


# ------------------------------------------------------------------ #
#  CORE LOADER                                                         #
# ------------------------------------------------------------------ #

def load_taxi(path: str, sample_size: int = 50000,
              random_seed: int = 42) -> pd.DataFrame:
    """
    Loads NYC Taxi CSV, cleans it, samples to sample_size rows.
    Handles the mixed-type warning on store_and_fwd_flag by
    specifying dtypes explicitly on import.

    Parameters
    ----------
    path        : path to the raw CSV file
    sample_size : number of rows to sample (default 50k)
    random_seed : for reproducibility

    Returns
    -------
    Clean DataFrame with FEATURE_COLS + TARGET_COL only.
    """
    dtype_spec = {
        "VendorID":             "Int64",
        "passenger_count":      "float64",
        "trip_distance":        "float64",
        "RatecodeID":           "float64",
        "store_and_fwd_flag":   "str",        # fixes DtypeWarning
        "PULocationID":         "Int64",
        "DOLocationID":         "Int64",
        "payment_type":         "Int64",
        "fare_amount":          "float64",
        "extra":                "float64",
        "mta_tax":              "float64",
        "tip_amount":           "float64",
        "tolls_amount":         "float64",
        "improvement_surcharge":"float64",
        "total_amount":         "float64",
        "congestion_surcharge": "float64",
    }

    df = pd.read_csv(path, dtype=dtype_spec, low_memory=False)

    # ── Clean ──────────────────────────────────────────────────────
    # Drop rows with missing values in our columns
    cols_needed = FEATURE_COLS + [TARGET_COL]
    df = df[cols_needed].dropna()

    # Filter outliers
    df = df[
        (df[TARGET_COL] >= FARE_MIN) &
        (df[TARGET_COL] <= FARE_MAX) &
        (df["trip_distance"] >= TRIP_MIN) &
        (df["trip_distance"] <= TRIP_MAX) &
        (df["passenger_count"] >= 1) &
        (df["passenger_count"] <= 6)
    ]

    # Sample
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)

    print(f"[DataLoader] Loaded {len(df)} rows from {path}")
    print(f"[DataLoader] Target mean={df[TARGET_COL].mean():.2f}, "
          f"std={df[TARGET_COL].std():.2f}")

    return df


# ------------------------------------------------------------------ #
#  TRAIN / TEST SPLIT                                                  #
# ------------------------------------------------------------------ #

def split_data(df: pd.DataFrame, test_size: float = 0.2,
               random_seed: int = 42):
    """
    Splits DataFrame into train/test sets.
    Returns X_train, X_test, y_train, y_test as numpy arrays.
    """
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------ #
#  SCALER                                                              #
# ------------------------------------------------------------------ #

def get_scaler(X_train: np.ndarray) -> StandardScaler:
    """
    Fits and returns a StandardScaler on training data.
    Store this scaler — same scaler must be used at inference time.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


# ------------------------------------------------------------------ #
#  STREAMING SIMULATOR                                                 #
# ------------------------------------------------------------------ #

class StreamSimulator:
    """
    Simulates a streaming data pipeline by yielding
    fixed-size batches from the dataset sequentially.
    Loops back to start when exhausted.

    Usage:
        stream = StreamSimulator(df, batch_size=500)
        for batch in stream:
            predictions = model.predict(batch[FEATURE_COLS])
    """

    def __init__(self, df: pd.DataFrame, batch_size: int = 500):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.current_idx = 0
        self.total_batches_served = 0

    def __iter__(self):
        return self

    def __next__(self) -> pd.DataFrame:
        start = self.current_idx
        end = start + self.batch_size

        if end > len(self.df):
            # Loop back
            self.current_idx = 0
            start = 0
            end = self.batch_size

        batch = self.df.iloc[start:end].copy()
        self.current_idx = end
        self.total_batches_served += 1
        return batch

    def reset(self):
        self.current_idx = 0
        self.total_batches_served = 0


# ------------------------------------------------------------------ #
#  DRIFT DATASET BUILDER                                               #
# ------------------------------------------------------------------ #

def build_drift_pair(path_2019: str, path_2020: str,
                     sample_size: int = 50000, random_seed: int = 42):
    """
    Loads both years and returns them as a (reference, drifted) pair.
    2019 = reference (healthy baseline)
    2020 = natural drift source (real distribution shift)

    This pair is used to validate that your drift detection (P1)
    correctly flags the 2020 data as drifted without any injection.
    Acts as a sanity check for the benchmark.
    """
    df_ref = load_taxi(path_2019, sample_size, random_seed)
    df_drift = load_taxi(path_2020, sample_size, random_seed)

    print(f"\n[DriftPair] 2019 trip_distance mean: {df_ref['trip_distance'].mean():.3f}")
    print(f"[DriftPair] 2020 trip_distance mean: {df_drift['trip_distance'].mean():.3f}")
    print(f"[DriftPair] 2019 fare_amount mean:   {df_ref['fare_amount'].mean():.3f}")
    print(f"[DriftPair] 2020 fare_amount mean:   {df_drift['fare_amount'].mean():.3f}")

    return df_ref, df_drift


# ------------------------------------------------------------------ #
#  SELF TEST
#  Update paths before running on your machine
#  python data_loader.py
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import os

    PATH_2019 = os.path.join("data", "yellow_tripdata_2019-01.csv")
    PATH_2020 = os.path.join("data", "yellow_tripdata_2020-01.csv")

    print("=" * 55)
    print("DataLoader Self-Test")
    print("=" * 55)

    # Test single load
    df = load_taxi(PATH_2019)
    assert len(df) == 50000, "Sample size mismatch"
    assert all(c in df.columns for c in FEATURE_COLS + [TARGET_COL]), \
        "Missing columns"
    assert df.isnull().sum().sum() == 0, "NaNs present after cleaning"
    print("\n✅ Single load: passed")

    # Test split
    X_train, X_test, y_train, y_test = split_data(df)
    assert X_train.shape[1] == len(FEATURE_COLS), "Feature count mismatch"
    assert len(X_train) + len(X_test) == 50000, "Split size mismatch"
    print(f"✅ Train/test split: {len(X_train)} train, {len(X_test)} test")

    # Test scaler
    scaler = get_scaler(X_train)
    X_scaled = scaler.transform(X_train)
    assert abs(X_scaled.mean()) < 0.01, "Scaler not working correctly"
    print("✅ Scaler: passed")

    # Test stream simulator
    stream = StreamSimulator(df, batch_size=500)
    batch = next(stream)
    assert len(batch) == 500, "Batch size mismatch"
    batch2 = next(stream)
    assert batch.index[0] != batch2.index[0], "Stream not advancing"
    print("✅ StreamSimulator: passed")

    # Test drift pair
    print("\n--- Drift Pair ---")
    df_ref, df_drift = build_drift_pair(PATH_2019, PATH_2020)
    assert len(df_ref) == 50000
    assert len(df_drift) == 50000
    print("✅ Drift pair: passed")

    print("\n✅ All DataLoader tests passed. Ready for pipeline integration.")