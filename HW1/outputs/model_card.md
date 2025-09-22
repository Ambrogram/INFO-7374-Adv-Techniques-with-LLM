# Heart Disease Classification Model Card

## Model Overview
- **Algorithm**: K-Nearest Neighbors (KNN)
- **Best K**: 21
- **Test Accuracy**: 0.595

## Dataset
- **Size**: 1000 samples, 12 features
- **Target**: Binary classification (0=no disease, 1=disease)
- **Numeric Features**: 6 (Age, RestingBP, Cholesterol, FastingBS, MaxHR...)
- **Categorical Features**: 5 (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope)

## Preprocessing
- **Numeric Features**: Median imputation + StandardScaler
- **Categorical Features**: Most frequent imputation + OneHotEncoder
- **Missing Values**: Handled with appropriate imputation strategies

## Train/Validation/Test Split
- **Train**: 72.0%
- **Validation**: 8.0%
- **Test**: 20.0%
- **Stratification**: Yes (on target variable)
- **Random State**: 7

## Model Selection
KNN models were evaluated with K=3, 9, 21:
- **K=3**: 0.450 validation accuracy
- **K=9**: 0.562 validation accuracy
- **K=21**: 0.588 validation accuracy

**Selected K=21** based on highest validation accuracy

## Performance
- **Final Test Accuracy**: 0.595

## Caveats
- Model performance may vary with different random seeds
- KNN is sensitive to feature scaling (addressed with StandardScaler)
- Class imbalance was detected and logged but not addressed
- Model assumes Euclidean distance is appropriate for this problem
