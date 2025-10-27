#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weather FFNN Training Script (GPU-optimized)

- 保留你的原始设计与增强版改动
- 新增并默认启用 GPU 训练（若可用），包含：
  * GPU 选择与显存增长（memory growth）
  * 可选 XLA 编译 (jit_compile)
  * 自动/可控混合精度 (mixed precision, float16 on GPU)
  * tf.data 输入管线 + prefetch (提升吞吐)
  * verbose=1 训练与评估显示进度
  * 可选 --debug-steps 用于快速验证流程（交作业时不要用）

- 结果导出：CSV + JSON
"""

from __future__ import annotations

import os
# 静音 TensorFlow 低层日志（必须在 import tensorflow 之前设置）
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gc
import json
import argparse
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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
    """verbosity: 0=WARNING, 1=INFO, 2=DEBUG."""
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ----------------------------- Reproducibility ------------------------------ #

def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ----------------------------- GPU Setup ----------------------------------- #

def setup_gpu(gpu_index: Optional[int], use_mixed_precision: Optional[bool]) -> None:
    """选择 GPU、开启显存增长，以及（可选）混合精度。"""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logging.warning("No GPU detected. Training will run on CPU.")
        return

    # 选择 GPU
    try:
        if gpu_index is None:
            tf.config.set_visible_devices(gpus[0], "GPU")
            which = 0
        else:
            tf.config.set_visible_devices(gpus[gpu_index], "GPU")
            which = gpu_index
    except Exception as e:
        logging.warning(f"Failed to set visible GPU (index={gpu_index}): {e}. Using default.")
        which = 0

    # 显存增长
    try:
        for gpu in tf.config.get_visible_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"Using GPU:{which} with memory growth enabled.")
    except Exception as e:
        logging.warning(f"Failed to enable memory growth: {e}")

    # 混合精度策略（默认：GPU 上自动启用；CPU 不启用）
    try:
        from tensorflow.keras import mixed_precision
        if use_mixed_precision is None:
            # 自动：如果是 NVIDIA GPU，基本都支持；保守策略也行
            policy_name = "mixed_float16"
        else:
            policy_name = "mixed_float16" if use_mixed_precision else "float32"
        mixed_precision.set_global_policy(policy_name)
        logging.info(f"Mixed precision policy set to: {mixed_precision.global_policy()}")
    except Exception as e:
        logging.warning(f"Failed to set mixed precision policy: {e}")


# ---------------------------- Data Preprocessing ---------------------------- #

TARGET_COLUMN = "RainTomorrow"

def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    logging.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    logging.info(f"Dataset shape: {df.shape}, columns: {list(df.columns)}")
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    y_raw = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    if y_raw.dtype == object:
        y = (
            y_raw.astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
        )
        if y.isna().any():
            y = pd.to_numeric(y_raw, errors="coerce")
    else:
        y = pd.to_numeric(y_raw, errors="coerce")

    if y.isna().any():
        non_null = ~y.isna()
        logging.warning(f"Dropping {int((~non_null).sum())} rows with invalid target values")
        X = X.loc[non_null]
        y = y.loc[non_null]

    y = y.astype(int)
    nunique = y.nunique()
    if nunique != 2:
        logging.warning(f"Target has {nunique} unique values after cleaning (expected 2).")

    return X, y


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >= 1.2
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn < 1.2


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
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

    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    # dtype：若启用 mixed_float16，输入仍可用 float32；TF 在内部处理
    X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
    X_test_processed = np.asarray(X_test_processed, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)

    logging.info(f"Train shape: X={X_train_processed.shape}, y={y_train.shape}")
    logging.info(f"Test  shape: X={X_test_processed.shape}, y={y_test.shape}")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


# ------------------------------- Model Building ----------------------------- #

def build_ffnn_model(input_dim: int, hidden_units: List[int], learning_rate: float, jit: bool) -> keras.Model:
    """构建 FFNN；当启用 mixed_float16 时，最后一层强制 float32 以避免数值不稳定。"""
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for units in hidden_units:
        model.add(layers.Dense(units, activation="relu"))
    # 若使用混合精度，建议输出层改为 float32
    out_dtype = "float32"
    model.add(layers.Dense(1, activation="sigmoid", dtype=out_dtype))

    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"], jit_compile=jit)
    return model


# --------------------------------- Training -------------------------------- #

@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float


def make_datasets(X_train, y_train, X_test, y_test, batch_size: int):
    """构建 tf.data 数据管线（shuffle+batch+prefetch）。"""
    # 10% 作为验证切片（从训练集中取前段，简单可控）
    val_size = max(1, int(0.1 * len(X_train)))
    X_val, y_val = X_train[:val_size], y_train[:val_size]
    X_tr,  y_tr  = X_train[val_size:], y_train[val_size:]

    ds_train = tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
    ds_train = ds_train.shuffle(min(len(X_tr), 10000)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    ds_val = ds_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    # 评估阶段可用更大 batch 加速（不影响训练配置）
    ds_test = ds_test.batch(max(256, batch_size)).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_test


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_units: List[int],
    config: TrainingConfig,
    seed: int,
    jit: bool,
    debug_steps: Optional[int] = None,
) -> Dict[str, float]:
    set_global_seed(seed)
    model = build_ffnn_model(
        input_dim=X_train.shape[1], hidden_units=hidden_units,
        learning_rate=config.learning_rate, jit=jit
    )

    ds_train, ds_val, ds_test = make_datasets(X_train, y_train, X_test, y_test, config.batch_size)

    logging.info(f"Training model: layers={hidden_units}, "
                 f"batch_size={config.batch_size}, epochs={config.epochs}, lr={config.learning_rate}, "
                 f"jit={jit}, debug_steps={debug_steps}")

    history = model.fit(
        ds_train,
        epochs=config.epochs,
        verbose=1,                         # 显示进度
        validation_data=ds_val,
        steps_per_epoch=debug_steps,       # None=全量；调试时可设置一个较小值
    )

    loss, accuracy = model.evaluate(ds_test, verbose=1)
    logging.info(f"Eval -> loss: {loss:.4f}, acc: {accuracy:.4f}")

    keras.backend.clear_session()
    del model, history, ds_train, ds_val, ds_test
    gc.collect()

    return {"loss": float(loss), "accuracy": float(accuracy)}


# ---------------------------------- Runner --------------------------------- #

def run_experiments(
    csv_path: str, seed: int, jit: bool, debug_steps: Optional[int]
) -> pd.DataFrame:
    set_global_seed(seed)
    df = load_dataset(csv_path)
    X_train, X_test, y_train, y_test, _ = preprocess_data(df, test_size=0.25, seed=seed)

    architectures: Dict[str, List[int]] = {
        "Model A (1x8)": [8],
        "Model B (8,5)": [8, 5],
    }
    configs: Dict[str, TrainingConfig] = {
        "bs32_ep10_lr1e-4": TrainingConfig(batch_size=32, epochs=10, learning_rate=1e-4),
        "bs4_ep30_lr1e-2":  TrainingConfig(batch_size=4,  epochs=30, learning_rate=1e-2),
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
                jit=jit,
                debug_steps=debug_steps,
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
    parser = argparse.ArgumentParser(description="Train FFNNs on weather dataset (GPU-optimized).")
    parser.add_argument("--data", type=str, default="weather.csv",
                        help="Path to CSV (default: weather.csv; will also try a common UUID fallback).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--outdir", type=str, default=".", help="Directory to save results (CSV/JSON).")
    parser.add_argument("-v", "--verbose", action="count", default=1,
                        help="Increase logging verbosity: -v (INFO, default), -vv (DEBUG).")
    parser.add_argument("--gpu-index", type=int, default=None, help="Select GPU index (default: first GPU).")
    parser.add_argument("--no-mixed-precision", action="store_true",
                        help="Disable mixed precision even on GPU.")
    parser.add_argument("--jit", action="store_true",
                        help="Enable XLA JIT compile for model.compile(jit_compile=True).")
    parser.add_argument("--debug-steps", type=int, default=None,
                        help="Limit steps_per_epoch for quick debugging (do NOT use for final grading).")
    args = parser.parse_args()

    setup_logging(args.verbose)
    set_global_seed(args.seed)

    # GPU & mixed precision
    use_mixed_precision = None if not args.no_mixed_precision else False
    setup_gpu(args.gpu_index, use_mixed_precision)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        args.data,
        os.path.join(script_dir, args.data),
        os.path.join(script_dir, "d5c82616-6d1d-49f8-8f7b-71214732bfcc.csv"),
    ]
    csv_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if csv_path is None:
        raise FileNotFoundError(f"Could not find CSV. Tried: {candidate_paths}")

    logging.info(f"Using CSV: {csv_path}")

    results = run_experiments(csv_path, seed=args.seed, jit=args.jit, debug_steps=args.debug_steps)

    # 输出
    print("\n=== Test Accuracy Results (higher is better) ===\n")
    print(results[["Architecture", "Config", "BatchSize", "Epochs", "LearningRate", "Test Accuracy"]]
          .to_string(index=False))

    print("\n=== Loss metrics (for reference) ===\n")
    print(results[["Architecture", "Config", "BatchSize", "Epochs", "LearningRate", "Test Loss"]]
          .to_string(index=False))

    os.makedirs(args.outdir, exist_ok=True)
    csv_out = os.path.join(args.outdir, "weather_ffnn_results.csv")
    json_out = os.path.join(args.outdir, "weather_ffnn_results.json")
    results.to_csv(csv_out, index=False)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    logging.info(f"Saved CSV to: {csv_out}")
    logging.info(f"Saved JSON to: {json_out}")

    best_row = results.loc[results["Test Accuracy"].idxmax()]
    print("\n=== Run Summary ===")
    print(f"Best Accuracy: {best_row['Test Accuracy']:.6f} | "
          f"Arch: {best_row['Architecture']} | Config: {best_row['Config']} | "
          f"bs={best_row['BatchSize']}, epochs={best_row['Epochs']}, lr={best_row['LearningRate']}")


if __name__ == "__main__":
    main()
