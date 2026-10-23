import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler
import joblib

from src.config import (
    NETFLOW_PARQUET_PATH,
    NETFLOW_FEATURE_COLS,
    ATTACK_CLASSES,
    CLASS_TO_IDX,
    SAVED_MODELS_DIR,
)

class NetFlowLoader:
    """
    DataLoader and preprocessor for NF-ToN-IoT-V2 network flow records.
    Provides stratified sampling across all 10 attack/benign categories.
    """
    def __init__(self, parquet_path: Optional[str] = None):
        self.parquet_path = str(parquet_path or NETFLOW_PARQUET_PATH)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def load_stratified_sample(
        self,
        samples_per_class: int = 10000,
        rare_class_boost: bool = True
    ) -> pd.DataFrame:
        """
        Loads a balanced/stratified sample from the 13.1M parquet dataset across all 10 classes.
        """
        if not os.path.exists(self.parquet_path):
            raise FileNotFoundError(f"Parquet dataset not found at {self.parquet_path}")

        dataset = ds.dataset(self.parquet_path, format="parquet")
        needed_cols = NETFLOW_FEATURE_COLS + ["Label", "Attack"]
        
        # Read dataset in chunks and build per-class pools
        class_dfs: Dict[str, List[pd.DataFrame]] = {cls_name: [] for cls_name in ATTACK_CLASSES}
        class_counts: Dict[str, int] = {cls_name: 0 for cls_name in ATTACK_CLASSES}
        
        scanner = dataset.scanner(columns=needed_cols, batch_size=200000)
        
        for batch in scanner.to_batches():
            chunk = batch.to_pandas()
            for cls_name in ATTACK_CLASSES:
                # If we still need samples for this class
                target_count = samples_per_class
                # For rare classes, take as many as available
                if rare_class_boost and cls_name in ["ransomware", "mitm", "backdoor"]:
                    target_count = 50000

                current_count = class_counts[cls_name]
                if current_count < target_count:
                    match = chunk[chunk["Attack"] == cls_name]
                    if len(match) > 0:
                        needed = target_count - current_count
                        selected = match.head(needed)
                        class_dfs[cls_name].append(selected)
                        class_counts[cls_name] += len(selected)

            # Check if all classes have enough samples or if rare classes have maxed out
            sufficient = True
            for cls_name in ATTACK_CLASSES:
                limit = 2500 if cls_name == "ransomware" else (5000 if cls_name == "mitm" else samples_per_class)
                if class_counts[cls_name] < limit:
                    sufficient = False
                    break
            if sufficient:
                break

        # Combine all collected pools
        combined_dfs = []
        for cls_name, dfs in class_dfs.items():
            if dfs:
                combined_dfs.append(pd.concat(dfs, ignore_index=True))

        df_all = pd.concat(combined_dfs, ignore_index=True)
        # Shuffle
        df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)
        return df_all

    def fit_transform_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fits scaler on features and returns (normalized_features, labels_multiclass, labels_binary).
        """
        # Clip very large values (e.g. 64-bit timestamps or throughput numbers) to avoid float32 overflow
        raw_vals = df[NETFLOW_FEATURE_COLS].values
        raw_vals = np.clip(raw_vals, -1e12, 1e12)
        X = raw_vals.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True

        y_multi = np.array([CLASS_TO_IDX.get(a, 0) for a in df["Attack"]], dtype=np.int64)
        y_binary = df["Label"].values.astype(np.int64)

        return X_scaled, y_multi, y_binary

    def transform_features(self, df_or_array) -> np.ndarray:
        """
        Transforms features using the fitted scaler for inference.
        """
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted yet. Fit or load scaler first.")
        if isinstance(df_or_array, pd.DataFrame):
            raw_vals = df_or_array[NETFLOW_FEATURE_COLS].values
        else:
            raw_vals = np.array(df_or_array)
        raw_vals = np.clip(raw_vals, -1e12, 1e12)
        X = raw_vals.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        return self.scaler.transform(X)

    def save_scaler(self, filepath: Optional[str] = None):
        path = filepath or str(SAVED_MODELS_DIR / "netflow_scaler.joblib")
        joblib.dump(self.scaler, path)

    def load_scaler(self, filepath: Optional[str] = None):
        path = filepath or str(SAVED_MODELS_DIR / "netflow_scaler.joblib")
        if os.path.exists(path):
            self.scaler = joblib.load(path)
            self.is_fitted = True
