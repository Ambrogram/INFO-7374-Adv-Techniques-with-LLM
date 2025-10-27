#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weather FFNN Training Script (enhanced)

- Uses your original architecture & pipeline, plus:
  * Clean logging (info-level progress; silenced TF C++ noise)
  * OneHotEncoder compatibility (sklearn >=1.2 sparse_output / <1.2 sparse)
  * Robust stratify guard (handles rare single-class edge case)
  * Keras session clear between runs to avoid memory creep
  * Results exported to CSV and JSON
  * CLI args (data path, seed, output dir), fallback to common UUID name
"""

from __future__ import annotations

import os
# Silence low-level TF logs before importing tensorflow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gc
import json
import argparse
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ----------------------------- Logging ------------------------------------- #

def setup_logging(verbosity: int = 1) -> None:
    """Configure root logger. verbosity: 0=WARNING, 1=INFO, 2=DEBUG."""
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ----------------------------- Reproducibility ------------------------------ #

def set_global_seed(seed: int = 42) -> None:
    """Set global random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------- Data Preprocessing ---------------------------- #

TARGET_COLUMN = "RainTomorrow"

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    logging.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    logging.info(f"Dataset shape: {df.shape}, columns: {list(df.columns)}")
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features X and binary target y; normalize target to {0,1}."""
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    y_raw = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # Normalize target to {0,1}
    if y_raw.dtype == object:
        y = (
            y_raw.astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
        )
        if y.isna().any():
            # fallback: coerce numerics
            y = pd.to_numeric(y_raw, errors="coerce")
    else:
        y = pd.to_numeric(y_raw, errors="coerce")

    if y.isna().any():
        # Drop rows where target couldn't be parsed
        non_null = ~y.isna()
        logging.warning(f"Dropping {int((~non_null).sum())} rows with invalid target values")
        X = X.loc[non_null]
        y = y.loc[non_null]

    # Ensure integer 0/1
    y = y.astype(int)
    # Safety: ensure binary
    nunique = y.nunique()
    if nunique != 2:
        logging.warning(f"Target has {nunique} unique values after cleaning (expected 2).")

    return X, y


def _make_ohe() -> OneHotEncoder:
    """Create OneHotEncoder with version compatibility."""
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a ColumnTransformer: 
       - Numerical: median imputation + standard scaling
       - Categorical: most frequent imputation + one-hot (ignore unknown)"""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    logging.info(f"Numeric features: {len(numeric_features)} | Categorical features: {len(categorical_features)}")

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_ohe()),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
    return preprocessor


def preprocess_data(
    df: pd.DataFrame, test_size: float = 0.25, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
    """Split data and fit/transform using the preprocessing pipeline."""
    X, y = split_features_target(df)

    stratify_arg = y if y.nunique() > 1 else None
    if stratify_arg is None:
        logging.warning("Stratify is disabled because the target has a single class after cleaning.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify_arg
    )

    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # If some sklearn version returns sparse matrices, densify
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    # Ensure numpy arrays with correct dtype
    X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
    X_test_processed = np.asarray(X_test_processed, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)

    logging.info(f"Train shape: X={X_train_processed.shape}, y={y_train.shape}")
    logging.info(f"Test  shape: X={X_test_processed.shape}, y={y_test.shape}")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


# ------------------------------- Model Building ----------------------------- #

def build_ffnn_model(input_dim: int, hidden_units: List[int], learning_rate: float) -> keras.Model:
    """Build a simple FFNN classifier with the given hidden units."""
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for units in hidden_units:
        model.add(layers.Dense(units, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


# --------------------------------- Training -------------------------------- #

@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_units: List[int],
    config: TrainingConfig,
    seed: int = 42,
) -> Dict[str, float]:
    """Train a model and return evaluation metrics on the test set."""
    set_global_seed(seed)
    model = build_ffnn_model(
        input_dim=X_train.shape[1], hidden_units=hidden_units, learning_rate=config.learning_rate
    )

    logging.info(f"Training model: layers={hidden_units}, "
                 f"batch_size={config.batch_size}, epochs={config.epochs}, lr={config.learning_rate}")

    # You can add callbacks here if desired (e.g., EarlyStopping)
    history = model.fit(
        X_train,
        y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        verbose=0,
        validation_split=0.1,
        shuffle=True,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    logging.info(f"Eval -> loss: {loss:.4f}, acc: {accuracy:.4f}")

    # Clean up TF graph/session to avoid memory accumulation
    keras.backend.clear_session()
    del model, history
    gc.collect()

    return {"loss": float(loss), "accuracy": float(accuracy)}


# ---------------------------------- Runner --------------------------------- #

def run_experiments(csv_path: str, seed: int = 42) -> pd.DataFrame:
    """Run all required experiments and return a DataFrame with results."""
    set_global_seed(seed)
    df = load_dataset(csv_path)
    X_train, X_test, y_train, y_test, _ = preprocess_data(df, test_size=0.25, seed=seed)

    # Architectures
    architectures: Dict[str, List[int]] = {
        "Model A (1x8)": [8],
        "Model B (8,5)": [8, 5],
    }

    # Training configurations
    configs: Dict[str, TrainingConfig] = {
        "bs32_ep10_lr1e-4": TrainingConfig(batch_size=32, epochs=10, learning_rate=1e-4),
        "bs4_ep30_lr1e-2": TrainingConfig(batch_size=4, epochs=30, learning_rate=1e-2),
    }

    records: List[Dict[str, object]] = []
    for arch_name, hidden_units in architectures.items():
        for cfg_name, cfg in configs.items():
            metrics = train_and_evaluate(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                hidden_units=hidden_units,
                config=cfg,
                seed=seed,
            )
            records.append(
                {
                    "Architecture": arch_name,
                    "Config": cfg_name,
                    "BatchSize": cfg.batch_size,
                    "Epochs": cfg.epochs,
                    "LearningRate": cfg.learning_rate,
                    "Test Accuracy": round(metrics["accuracy"], 6),
                    "Test Loss": round(metrics["loss"], 6),
                }
            )

    results_df = pd.DataFrame.from_records(records)
    results_df = results_df.sort_values(["Architecture", "Config"]).reset_index(drop=True)
    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FFNNs on weather dataset.")
    parser.add_argument("--data", type=str, default="weather.csv",
                        help="Path to CSV (default: weather.csv, will try a common UUID fallback).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--outdir", type=str, default=".",
                        help="Directory to save results CSV/JSON (default: current dir).")
    parser.add_argument("-v", "--verbose", action="count", default=1,
                        help="Increase logging verbosity: -v (INFO, default), -vv (DEBUG).")
    args = parser.parse_args()

    setup_logging(args.verbose)
    set_global_seed(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        args.data,
        os.path.join(script_dir, args.data),
        os.path.join(script_dir, "d5c82616-6d1d-49f8-8f7b-71214732bfcc.csv"),  # common uploaded UUID fallback
    ]

    csv_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            csv_path = p
            break
    if csv_path is None:
        raise FileNotFoundError(f"Could not find CSV. Tried: {candidate_paths}")

    logging.info(f"Using CSV: {csv_path}")

    results = run_experiments(csv_path, seed=args.seed)

    # Pretty print results
    print("\n=== Test Accuracy Results (higher is better) ===\n")
    print(results[["Architecture", "Config", "BatchSize", "Epochs", "LearningRate", "Test Accuracy"]]
          .to_string(index=False))

    print("\n=== Loss metrics (for reference) ===\n")
    print(results[["Architecture", "Config", "BatchSize", "Epochs", "LearningRate", "Test Loss"]]
          .to_string(index=False))

    # Ensure outdir exists
    os.makedirs(args.outdir, exist_ok=True)
    csv_out = os.path.join(args.outdir, "weather_ffnn_results.csv")
    json_out = os.path.join(args.outdir, "weather_ffnn_results.json")

    results.to_csv(csv_out, index=False)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    logging.info(f"Saved CSV to: {csv_out}")
    logging.info(f"Saved JSON to: {json_out}")

    # Small run summary
    best_row = results.loc[results["Test Accuracy"].idxmax()]
    print("\n=== Run Summary ===")
    print(f"Best Accuracy: {best_row['Test Accuracy']:.6f} | "
          f"Arch: {best_row['Architecture']} | Config: {best_row['Config']} | "
          f"bs={best_row['BatchSize']}, epochs={best_row['Epochs']}, lr={best_row['LearningRate']}")


if __name__ == "__main__":
    main()
