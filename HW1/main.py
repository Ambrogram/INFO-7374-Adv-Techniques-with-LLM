#!/usr/bin/env python3
"""
Heart Disease Modeling & Clustering Pipeline

A production-quality Python script that loads a heart-disease dataset,
performs KNN modeling with evaluation on prescribed splits, and runs
K-Means clustering with 2D visualization.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def load_data(path: str) -> pd.DataFrame:
    """Load CSV into DataFrame with validation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    logging.info(f"Loaded data with shape: {df.shape}")
    return df


def preprocess_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Ensure target is binary 0/1, handling Yes/No strings and logging imbalance."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    # Convert common string encodings
    if df[target_col].dtype == 'object':
        lower = df[target_col].astype(str).str.lower()
        mapped = lower.map({'yes': 1, 'y': 1, 'true': 1, '1': 1, 'no': 0, 'n': 0, 'false': 0, '0': 0})
        df[target_col] = mapped.where(mapped.notna(), pd.to_numeric(df[target_col], errors='coerce'))

    # Coerce to int 0/1
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)

    class_counts = df[target_col].value_counts()
    if not class_counts.empty and (class_counts.min() / len(df) < 0.2):
        logging.warning("Class imbalance detected: minority class ratio = %.3f", class_counts.min() / len(df))
    logging.info(f"Target distribution: {dict(class_counts)}")
    return df


def split_data(X: pd.DataFrame, y: pd.Series, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split into 72/8/20 using stratification and fixed random_state."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.10, stratify=y_tmp, random_state=random_state, shuffle=True
    )

    total = len(X)
    logging.info("Data split sizes:")
    logging.info("  Train: %d (%.3f)", len(X_train), len(X_train) / total)
    logging.info("  Validation: %d (%.3f)", len(X_val), len(X_val) / total)
    logging.info("  Test: %d (%.3f)", len(X_test), len(X_test) / total)
    for name, y_split in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
        logging.info("  %s class distribution: %s", name, dict(y_split.value_counts(normalize=True).sort_index()))
    return X_train, X_val, X_test, y_train, y_val, y_test


def identify_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Auto-detect numeric vs categorical columns by dtype."""
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    logging.info("Feature types identified:")
    logging.info("  Numeric columns (%d): %s", len(numeric_cols), numeric_cols)
    logging.info("  Categorical columns (%d): %s", len(categorical_cols), categorical_cols)
    return numeric_cols, categorical_cols


def build_knn_pipeline(k: int, numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    """Build KNN Pipeline with ColumnTransformer preprocessing."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2))
    ])
    return pipeline


def evaluate_knn_candidates(X_train: pd.DataFrame, X_val: pd.DataFrame,
                            y_train: pd.Series, y_val: pd.Series,
                            numeric_cols: List[str], categorical_cols: List[str]) -> Dict[str, float]:
    """Evaluate K in {3,9,21} and return validation accuracies."""
    k_values = [3, 9, 21]
    accuracies: Dict[str, float] = {}
    for k in k_values:
        pipe = build_knn_pipeline(k, numeric_cols, categorical_cols)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        acc = accuracy_score(y_val, preds)
        accuracies[str(k)] = acc
        logging.info("K=%d validation accuracy: %.3f", k, acc)
    return accuracies


def train_final_knn(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                    y_train: pd.Series, y_val: pd.Series, y_test: pd.Series,
                    best_k: int, numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[Pipeline, float]:
    """Retrain on train+val and evaluate on test."""
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    pipe = build_knn_pipeline(best_k, numeric_cols, categorical_cols)
    pipe.fit(X_train_val, y_train_val)
    test_acc = accuracy_score(y_test, pipe.predict(X_test))
    logging.info("Final model (K=%d) test accuracy: %.3f", best_k, test_acc)
    return pipe, test_acc


def run_kmeans_and_plot(X_all: pd.DataFrame,
                        numeric_cols: List[str], categorical_cols: List[str],
                        output_dir: str) -> Optional[str]:
    """Fit KMeans(n_clusters=2) on all features and scatter MaxHR vs Age by cluster."""
    missing = [c for c in ['MaxHR', 'Age'] if c not in X_all.columns]
    if missing:
        logging.warning(f"Missing columns for clustering plot: {missing}. Skipping plot.")
        return None

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    X_proc = preprocessor.fit_transform(X_all)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_proc)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_all['MaxHR'], X_all['Age'], c=clusters, cmap='viridis', alpha=0.7)
    plt.xlabel('MaxHR')
    plt.ylabel('Age')
    plt.title('K-Means Clustering: MaxHR vs Age')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'cluster_scatter_MaxHR_vs_Age.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Cluster plot saved: {plot_path}")
    return plot_path


def save_metrics(metrics: Dict, output_dir: str) -> str:
    """Write metrics.json to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'metrics.json')
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Metrics saved: {path}")
    return path


def write_model_card(metrics: Dict, output_dir: str, data_shape: Tuple[int, int],
                    numeric_cols: List[str], categorical_cols: List[str]) -> str:
    """Create a concise model card describing data, preprocessing, splits and metrics."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'model_card.md')
    with open(path, 'w') as f:
        f.write("# Heart Disease Classification Model Card\n\n")
        f.write("## Model Overview\n")
        f.write(f"- **Algorithm**: K-Nearest Neighbors (KNN)\n")
        f.write(f"- **Best K**: {metrics['best_k']}\n")
        f.write(f"- **Test Accuracy**: {metrics['test_accuracy']:.3f}\n\n")

        f.write("## Dataset\n")
        f.write(f"- **Size**: {data_shape[0]} samples, {data_shape[1]} columns\n")
        f.write("- **Target**: Binary classification (0=no disease, 1=disease)\n")
        f.write(f"- **Numeric Features**: {len(numeric_cols)} ({', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''})\n")
        f.write(f"- **Categorical Features**: {len(categorical_cols)} ({', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''})\n\n")

        f.write("## Preprocessing\n")
        f.write("- Numeric: Median imputation + StandardScaler\n")
        f.write("- Categorical: Most frequent imputation + OneHotEncoder(handle_unknown='ignore')\n\n")

        f.write("## Train/Validation/Test Split\n")
        f.write(f"- Train: {metrics['split']['train']:.0%}\n")
        f.write(f"- Validation: {metrics['split']['val']:.0%}\n")
        f.write(f"- Test: {metrics['split']['test']:.0%}\n")
        f.write("- Stratified on target, shuffle=True\n")
        f.write(f"- Random State: {metrics['random_state']}\n\n")

        f.write("## Model Selection\n")
        f.write("Validation accuracies for K in {3, 9, 21}:\n")
        for k, acc in metrics['validation_accuracy'].items():
            f.write(f"- K={k}: {acc:.3f}\n")
        f.write(f"\nSelected K={metrics['best_k']} (highest validation accuracy; tie-break favors smaller K if tied)\n\n")

        f.write("## Performance\n")
        f.write(f"- Final Test Accuracy: {metrics['test_accuracy']:.3f}\n\n")

        f.write("## Caveats\n")
        f.write("- Accuracy reported despite potential class imbalance (warned in logs)\n")
        f.write("- KNN sensitive to scaling; handled by StandardScaler\n")
        f.write("- Euclidean distance assumption may not be optimal for all datasets\n")
    logging.info(f"Model card saved: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description='Heart Disease Modeling & Clustering Pipeline')
    parser.add_argument('--data', required=True, help='Path to CSV data file')
    parser.add_argument('--target', default='HeartDisease', help='Target column name (default: HeartDisease)')
    parser.add_argument('--id-cols', nargs='*', default=[], help='ID columns to exclude from features')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility (default: 42)')
    parser.add_argument('--out-dir', default='./outputs', help='Output directory (default: ./outputs)')
    args = parser.parse_args()

    setup_logging()
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        logging.info('Loading data...')
        df = load_data(args.data)
        logging.info('Preprocessing target column...')
        df = preprocess_target(df, args.target)

        feature_cols = [c for c in df.columns if c not in [args.target] + args.id_cols]
        X = df[feature_cols]
        y = df[args.target]
        logging.info(f"Using {len(feature_cols)} features: {feature_cols}")

        numeric_cols, categorical_cols = identify_feature_types(X)

        logging.info('Splitting data...')
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, args.random_state)

        logging.info('Evaluating KNN candidates...')
        val_acc = evaluate_knn_candidates(X_train, X_val, y_train, y_val, numeric_cols, categorical_cols)

        # Select best K (highest accuracy; tie-break on smaller K)
        best_k = min(val_acc.keys(), key=lambda k: (-val_acc[k], int(k)))
        best_k_int = int(best_k)

        print("\nValidation Accuracies:")
        for k in [3, 9, 21]:
            print(f"- K={k}   : {val_acc[str(k)]:.3f}")
        print(f"Best K = {best_k_int} (highest validation accuracy; tie-break favors smaller K if tied)")

        logging.info(f'Training final model with K={best_k_int}...')
        _, test_accuracy = train_final_knn(X_train, X_val, X_test, y_train, y_val, y_test, best_k_int, numeric_cols, categorical_cols)
        print(f"Test Accuracy (K={best_k_int} trained on train+val): {test_accuracy:.3f}")

        logging.info('Running K-Means clustering...')
        plot_path = run_kmeans_and_plot(X, numeric_cols, categorical_cols, args.out_dir)
        if plot_path:
            print(f"Cluster plot saved: {plot_path}")

        metrics = {
            "random_state": args.random_state,
            "split": {"train": 0.72, "val": 0.08, "test": 0.20},
            "validation_accuracy": val_acc,
            "best_k": best_k_int,
            "test_accuracy": test_accuracy
        }
        metrics_path = save_metrics(metrics, args.out_dir)
        print(f"Metrics saved: {metrics_path}")

        model_card_path = write_model_card(metrics, args.out_dir, df.shape, numeric_cols, categorical_cols)
        print(f"Model card saved: {model_card_path}")

        logging.info('Pipeline completed successfully!')
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Heart Disease Modeling & Clustering Pipeline

A production-quality Python script that loads a heart-disease dataset,
performs KNN modeling with evaluation on prescribed splits, and runs
K-Means clustering with 2D visualization.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def load_data(path: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: If there's an error loading the file
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        raise Exception(f"Error loading data from {path}: {str(e)}")


def preprocess_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Preprocess the target column to ensure it's binary (0/1).
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        
    Returns:
        DataFrame with preprocessed target column
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Handle different target formats
    target_values = df[target_col].unique()
    logging.info(f"Target column '{target_col}' has unique values: {target_values}")
    
    if df[target_col].dtype == 'object':
        # Handle string values like "Yes"/"No"
        if set(target_values).issubset({'Yes', 'No', 'yes', 'no', 'Y', 'N', 'y', 'n'}):
            df[target_col] = df[target_col].str.lower().map({'yes': 1, 'y': 1, 'no': 0, 'n': 0})
            logging.info("Converted string target values to binary")
        else:
            # Try to convert to numeric first
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    
    # Ensure binary values
    df[target_col] = df[target_col].astype(int)
    
    # Check for class imbalance
    class_counts = df[target_col].value_counts()
    minority_ratio = class_counts.min() / len(df)
    if minority_ratio < 0.2:
        logging.warning(f"Class imbalance detected: minority class ratio = {minority_ratio:.3f}")
    
    logging.info(f"Target distribution: {dict(class_counts)}")
    return df


def split_data(X: pd.DataFrame, y: pd.Series, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train/validation/test sets with exact ratios.
    
    Args:
        X: Feature matrix
        y: Target vector
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: 80% (train+val) / 20% (test)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state, shuffle=True
    )
    
    # Second split: 90% (train) / 10% (val) of the 80%
    # This gives us 72% train / 8% val / 20% test
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.10, stratify=y_tmp, random_state=random_state, shuffle=True
    )
    
    # Log actual split sizes
    total_size = len(X)
    logging.info(f"Data split sizes:")
    logging.info(f"  Train: {len(X_train)} ({len(X_train)/total_size:.3f})")
    logging.info(f"  Validation: {len(X_val)} ({len(X_val)/total_size:.3f})")
    logging.info(f"  Test: {len(X_test)} ({len(X_test)/total_size:.3f})")
    
    # Log class distributions
    for split_name, y_split in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
        class_dist = y_split.value_counts(normalize=True).sort_index()
        logging.info(f"  {split_name} class distribution: {dict(class_dist)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def identify_feature_types(df: pd.DataFrame, id_cols: List[str]) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical columns.
    
    Args:
        df: Input DataFrame
        id_cols: List of ID columns to exclude
        
    Returns:
        Tuple of (numeric_columns, categorical_columns)
    """
    feature_cols = [col for col in df.columns if col not in id_cols]
    
    numeric_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    logging.info(f"Feature types identified:")
    logging.info(f"  Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    logging.info(f"  Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    return numeric_cols, categorical_cols


def build_knn_pipeline(k: int, numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    """
    Build KNN pipeline with preprocessing.
    
    Args:
        k: Number of neighbors for KNN
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        
    Returns:
        Configured Pipeline
    """
    # Create preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    
    # Create full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2))
    ])
    
    return pipeline


def evaluate_knn_candidates(X_train: pd.DataFrame, X_val: pd.DataFrame, 
                          y_train: pd.Series, y_val: pd.Series,
                          numeric_cols: List[str], categorical_cols: List[str]) -> Dict[str, float]:
    """
    Evaluate KNN models with different K values.
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training targets
        y_val: Validation targets
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        
    Returns:
        Dictionary mapping K values to validation accuracies
    """
    k_values = [3, 9, 21]
    validation_accuracies = {}
    
    for k in k_values:
        pipeline = build_knn_pipeline(k, numeric_cols, categorical_cols)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        validation_accuracies[str(k)] = accuracy
        logging.info(f"K={k} validation accuracy: {accuracy:.3f}")
    
    return validation_accuracies


def train_final_knn(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_val: pd.Series, y_test: pd.Series,
                   best_k: int, numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[Pipeline, float]:
    """
    Train final KNN model on train+val and evaluate on test set.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        y_train: Training targets
        y_val: Validation targets
        y_test: Test targets
        best_k: Best K value to use
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        
    Returns:
        Tuple of (trained_pipeline, test_accuracy)
    """
    # Combine train and validation sets
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    
    # Train final model
    pipeline = build_knn_pipeline(best_k, numeric_cols, categorical_cols)
    pipeline.fit(X_train_val, y_train_val)
    
    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    logging.info(f"Final model (K={best_k}) test accuracy: {test_accuracy:.3f}")
    
    return pipeline, test_accuracy


def run_kmeans_and_plot(X_all: pd.DataFrame, y_all: pd.Series, 
                       numeric_cols: List[str], categorical_cols: List[str],
                       output_dir: str) -> Optional[str]:
    """
    Run K-Means clustering and create scatter plot.
    
    Args:
        X_all: All features (train+val+test)
        y_all: All targets
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        output_dir: Output directory for the plot
        
    Returns:
        Path to saved plot file or None if plot couldn't be created
    """
    # Check if required columns exist
    plot_cols = ['MaxHR', 'Age']
    missing_cols = [col for col in plot_cols if col not in X_all.columns]
    
    if missing_cols:
        logging.warning(f"Missing columns for clustering plot: {missing_cols}. Skipping plot.")
        return None
    
    # Preprocess features for clustering (same as KNN)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    
    # Fit and transform all data
    X_processed = preprocessor.fit_transform(X_all)
    
    # Run K-Means
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_processed)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_all['MaxHR'], X_all['Age'], c=clusters, cmap='viridis', alpha=0.7)
    plt.xlabel('MaxHR')
    plt.ylabel('Age')
    plt.title('K-Means Clustering: MaxHR vs Age')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = os.path.join(output_dir, 'cluster_scatter_MaxHR_vs_Age.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Cluster plot saved: {plot_path}")
    return plot_path


def save_metrics(metrics: Dict, output_dir: str) -> str:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary containing metrics
        output_dir: Output directory
        
    Returns:
        Path to saved metrics file
    """
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logging.info(f"Metrics saved: {metrics_path}")
    return metrics_path


def write_model_card(metrics: Dict, output_dir: str, data_shape: Tuple[int, int], 
                    numeric_cols: List[str], categorical_cols: List[str]) -> str:
    """
    Write model card markdown file.
    
    Args:
        metrics: Dictionary containing metrics
        output_dir: Output directory
        data_shape: Shape of the original dataset
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        
    Returns:
        Path to saved model card file
    """
    model_card_path = os.path.join(output_dir, 'model_card.md')
    
    with open(model_card_path, 'w') as f:
        f.write("# Heart Disease Classification Model Card\n\n")
        f.write("## Model Overview\n")
        f.write(f"- **Algorithm**: K-Nearest Neighbors (KNN)\n")
        f.write(f"- **Best K**: {metrics['best_k']}\n")
        f.write(f"- **Test Accuracy**: {metrics['test_accuracy']:.3f}\n\n")
        
        f.write("## Dataset\n")
        f.write(f"- **Size**: {data_shape[0]} samples, {data_shape[1]} features\n")
        f.write(f"- **Target**: Binary classification (0=no disease, 1=disease)\n")
        f.write(f"- **Numeric Features**: {len(numeric_cols)} ({', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''})\n")
        f.write(f"- **Categorical Features**: {len(categorical_cols)} ({', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''})\n\n")
        
        f.write("## Preprocessing\n")
        f.write("- **Numeric Features**: Median imputation + StandardScaler\n")
        f.write("- **Categorical Features**: Most frequent imputation + OneHotEncoder\n")
        f.write("- **Missing Values**: Handled with appropriate imputation strategies\n\n")
        
        f.write("## Train/Validation/Test Split\n")
        f.write(f"- **Train**: {metrics['split']['train']:.1%}\n")
        f.write(f"- **Validation**: {metrics['split']['val']:.1%}\n")
        f.write(f"- **Test**: {metrics['split']['test']:.1%}\n")
        f.write("- **Stratification**: Yes (on target variable)\n")
        f.write(f"- **Random State**: {metrics['random_state']}\n\n")
        
        f.write("## Model Selection\n")
        f.write("KNN models were evaluated with K=3, 9, 21:\n")
        for k, acc in metrics['validation_accuracy'].items():
            f.write(f"- **K={k}**: {acc:.3f} validation accuracy\n")
        f.write(f"\n**Selected K={metrics['best_k']}** based on highest validation accuracy\n\n")
        
        f.write("## Performance\n")
        f.write(f"- **Final Test Accuracy**: {metrics['test_accuracy']:.3f}\n\n")
        
        f.write("## Caveats\n")
        f.write("- Model performance may vary with different random seeds\n")
        f.write("- KNN is sensitive to feature scaling (addressed with StandardScaler)\n")
        f.write("- Class imbalance was detected and logged but not addressed\n")
        f.write("- Model assumes Euclidean distance is appropriate for this problem\n")
    
    logging.info(f"Model card saved: {model_card_path}")
    return model_card_path


def main():
    """Main function to run the heart disease modeling pipeline."""
    parser = argparse.ArgumentParser(description='Heart Disease Modeling & Clustering Pipeline')
    parser.add_argument('--data', required=True, help='Path to CSV data file')
    parser.add_argument('--target', default='HeartDisease', help='Target column name (default: HeartDisease)')
    parser.add_argument('--id-cols', nargs='*', default=[], help='ID columns to exclude from features')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility (default: 42)')
    parser.add_argument('--out-dir', default='./outputs', help='Output directory (default: ./outputs)')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    os.makedirs(args.out_dir, exist_ok=True)
    
    try:
        # Load and preprocess data
        logging.info("Loading data...")
        df = load_data(args.data)
        
        logging.info("Preprocessing target column...")
        df = preprocess_target(df, args.target)
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in [args.target] + args.id_cols]
        X = df[feature_cols]
        y = df[args.target]
        
        logging.info(f"Using {len(feature_cols)} features: {feature_cols}")
        
        # Identify feature types
        numeric_cols, categorical_cols = identify_feature_types(X, args.id_cols)
        
        # Split data
        logging.info("Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, args.random_state)
        
        # Evaluate KNN candidates
        logging.info("Evaluating KNN candidates...")
        validation_accuracies = evaluate_knn_candidates(X_train, X_val, y_train, y_val, numeric_cols, categorical_cols)
        
        # Select best K
        best_k = min(validation_accuracies.keys(), key=lambda k: (-validation_accuracies[k], int(k)))
        best_k = int(best_k)
        
        print(f"\nValidation Accuracies:")
        for k in [3, 9, 21]:
            print(f"- K={k}   : {validation_accuracies[str(k)]:.3f}")
        
        print(f"Best K = {best_k} (highest validation accuracy; tie-break favors smaller K if tied)")
        
        # Train final model
        logging.info(f"Training final model with K={best_k}...")
        final_model, test_accuracy = train_final_knn(
            X_train, X_val, X_test, y_train, y_val, y_test, 
            best_k, numeric_cols, categorical_cols
        )
        
        print(f"Test Accuracy (K={best_k} trained on train+val): {test_accuracy:.3f}")
        
        # Run K-Means clustering and create plot
        logging.info("Running K-Means clustering...")
        plot_path = run_kmeans_and_plot(X, y, numeric_cols, categorical_cols, args.out_dir)
        if plot_path:
            print(f"Cluster plot saved: {plot_path}")
        
        # Save metrics
        metrics = {
            "random_state": args.random_state,
            "split": {"train": 0.72, "val": 0.08, "test": 0.20},
            "validation_accuracy": validation_accuracies,
            "best_k": best_k,
            "test_accuracy": test_accuracy
        }
        
        metrics_path = save_metrics(metrics, args.out_dir)
        print(f"Metrics saved: {metrics_path}")
        
        # Write model card
        model_card_path = write_model_card(metrics, args.out_dir, df.shape, numeric_cols, categorical_cols)
        print(f"Model card saved: {model_card_path}")
        
        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()


