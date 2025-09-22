# Heart Disease Modeling & Clustering Pipeline

A production-quality Python script that loads a heart-disease dataset, performs KNN modeling with evaluation on prescribed splits, and runs K-Means clustering with 2D visualization.

## Features

- **Stratified Data Splitting**: 72% train, 8% validation, 20% test
- **KNN Classification**: Evaluates K=3, 9, 21 and selects best based on validation accuracy
- **K-Means Clustering**: 2D visualization using MaxHR vs Age
- **Robust Preprocessing**: Handles missing values and mixed data types
- **Comprehensive Outputs**: Metrics JSON, model card, and cluster visualization

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python main.py --data path/to/heart_disease.csv
```

### Advanced Usage
```bash
python main.py --data data/heart.csv --id-cols PatientID RecordID --random-state 7 --out-dir results
```

### CLI Options

- `--data PATH` (required): Path to CSV data file
- `--target COLUMN` (default: HeartDisease): Target column name
- `--id-cols COL1 COL2 ...` (optional): ID columns to exclude from features
- `--random-state INT` (default: 42): Random state for reproducibility
- `--out-dir PATH` (default: ./outputs): Output directory

## Output Files

The pipeline generates the following outputs in the specified directory:

1. **metrics.json**: Contains validation accuracies, best K selection, and test accuracy
2. **cluster_scatter_MaxHR_vs_Age.png**: 2D scatter plot of K-Means clusters (if MaxHR and Age columns exist)
3. **model_card.md**: Human-readable model documentation

## Example Output

```
Validation Accuracies:
- K=3   : 0.842
- K=9   : 0.856
- K=21  : 0.848
Best K = 9 (highest validation accuracy; tie-break favors smaller K if tied)
Test Accuracy (K=9 trained on train+val): 0.861
Cluster plot saved: outputs/cluster_scatter_MaxHR_vs_Age.png
Metrics saved: outputs/metrics.json
Model card saved: outputs/model_card.md
```

## Data Requirements

- **Target Column**: Binary classification (0=no disease, 1=disease)
- **Feature Columns**: All other columns except optional ID columns
- **Plotting Columns**: MaxHR and Age (for cluster visualization)
- **Data Types**: Automatically detects numeric vs categorical features

## Preprocessing

- **Numeric Features**: Median imputation + StandardScaler
- **Categorical Features**: Most frequent imputation + OneHotEncoder
- **Missing Values**: Handled with appropriate imputation strategies
- **Target Conversion**: Automatically converts string values (Yes/No) to binary (1/0)

## Model Selection

The pipeline evaluates three KNN models (K=3, 9, 21) and selects the one with highest validation accuracy. In case of ties, the smaller K value is preferred.

## Error Handling

- Validates file existence and target column presence
- Warns about class imbalance and missing plotting columns
- Gracefully handles missing values and data type issues
- Provides clear error messages for common issues

## Dependencies

- pandas >= 1.5.0, < 2.0.0
- numpy >= 1.21.0, < 2.0.0
- scikit-learn >= 1.1.0, < 2.0.0
- matplotlib >= 3.5.0, < 4.0.0

## Testing

The pipeline has been tested with:
- Sample heart disease dataset (1000 samples, 12 features)
- Different random states
- ID column exclusion
- Missing plotting columns
- Error conditions (non-existent files, invalid target columns)

## Code Quality

- Type hints throughout
- Comprehensive docstrings
- PEP8 compliant
- Clean function decomposition
- Extensive logging
- Robust error handling
