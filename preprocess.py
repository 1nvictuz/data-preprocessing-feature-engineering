
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import sklearn

# Project configuration
PROJECT = Path(__file__).resolve().parent
INPUT_FILE = PROJECT / "example_dataset.csv"
OUTPUT = PROJECT / "output"
OUTPUT.mkdir(exist_ok=True)

RANDOM_SEED = 42

# 1. Load dataset
df = pd.read_csv(INPUT_FILE)

print("=== DATA PREPROCESSING ASSIGNMENT ===")
print("Original dataset shape:", df.shape)

# 2. Check data quality
missing = df.isna().sum().to_dict()
duplicates = int(df.duplicated().sum())

print("\nMissing values:")
print(df.isna().sum())
print("\nDuplicate records:", duplicates)

# Confirm the required numerical features exist
features = [f"feature_{i}" for i in range(1, 6)]

if not set(features).issubset(df.columns):
    raise ValueError("Required feature columns are missing.")

# No target labels exist in this dataset
print("\nLabel check: No label column exists.")
print("Classification baseline: Not applicable.")

# Remove exact duplicates before splitting
clean_df = df.drop_duplicates().copy()

# 3. Split before learning preprocessing rules
train_df, eval_df = train_test_split(
    clean_df,
    test_size=0.20,
    random_state=RANDOM_SEED,
    shuffle=True
)

# 4. Prepare numeric features
train_numeric = train_df[features].apply(
    pd.to_numeric, errors="coerce"
)

eval_numeric = eval_df[features].apply(
    pd.to_numeric, errors="coerce"
)

# Learn imputation values from training only
imputer = SimpleImputer(strategy="mean")
train_imputed = imputer.fit_transform(train_numeric)
eval_imputed = imputer.transform(eval_numeric)

# Learn scaling values from training only
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_imputed)
eval_scaled = scaler.transform(eval_imputed)

train_processed = pd.DataFrame(
    train_scaled, columns=features
)

eval_processed = pd.DataFrame(
    eval_scaled, columns=features
)

# 5. Feature engineering
def engineer_features(data):
    result = data.copy()

    result["feature_1_x_feature_2"] = (
        result["feature_1"] * result["feature_2"]
    )

    result["feature_1_plus_feature_2"] = (
        result["feature_1"] + result["feature_2"]
    )

    result["feature_average"] = (
        result[features].mean(axis=1)
    )

    return result

train_processed = engineer_features(train_processed)
eval_processed = engineer_features(eval_processed)

# 6. Save processed datasets
train_processed.to_csv(
    OUTPUT / "train_preprocessed.csv", index=False
)

eval_processed.to_csv(
    OUTPUT / "evaluation_preprocessed.csv", index=False
)

# 7. Save report
report = {
    "original_rows": len(df),
    "original_columns": len(df.columns),
    "duplicate_records": duplicates,
    "missing_values": {
        key: int(value) for key, value in missing.items()
    },
    "has_target_labels": False,
    "label_validation": "Not applicable: no label column",
    "baseline": "Not applicable: no target variable",
    "training_rows": len(train_processed),
    "evaluation_rows": len(eval_processed),
    "split_ratio": "80/20",
    "random_seed": RANDOM_SEED,
    "imputation": "Mean fitted on training data only",
    "scaling": "StandardScaler fitted on training data only",
    "engineered_features": [
        "feature_1_x_feature_2",
        "feature_1_plus_feature_2",
        "feature_average"
    ],
    "python_version": sys.version.split()[0],
    "pandas_version": pd.__version__,
    "numpy_version": np.__version__,
    "sklearn_version": sklearn.__version__,
    "limitations": (
        "No target labels exist. Predictive accuracy and "
        "classification baseline cannot be calculated."
    )
}

with open(OUTPUT / "report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

print("\n=== RESULTS ===")
print("Training records:", len(train_processed))
print("Evaluation records:", len(eval_processed))
print("Engineered features: 3")
print("Total processed features:", len(train_processed.columns))

print("\nFiles saved successfully!")
