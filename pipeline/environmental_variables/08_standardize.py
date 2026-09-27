"""
standardize_environmental_predictors.py

Standardize the continuous environmental predictors using z-scores:

    z = (value - column mean) / column standard deviation

The original input file is not modified. The standardized data are written to
environmental_predictors_standardized.csv.

NMDS1 and NMDS2 are response variables, so they are retained but not
standardized. Identifier and categorical columns are also retained unchanged.
Because angle_degrees is circular, it is converted to light_angle_sin and
light_angle_cos instead of being standardized directly.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
OUTPUT = DATA / "environmental_variables" / "output"

INPUT_FILE = OUTPUT / "environmental_predictors.csv"
OUTPUT_FILE = OUTPUT / "environmental_predictors_standardized.csv"


# ============================================================
# Columns not treated as continuous predictors
# ============================================================

# Responses are kept on their original NMDS scales.
RESPONSE_COLS = {
    "NMDS1",
    "NMDS2",
}

# Identifiers, labels, and categorical codes are copied unchanged.
NON_PREDICTOR_COLS = {
    "site",
    "name",
    "id",
    "utcOffset",
    "land cover class code",
    "landcover_class_code",
    "landcover_class_point",
    "nearest_river",
    "nearest_highway",
}

# Raw angles must not be z-standardized because -179 degrees and +179 degrees
# represent almost the same direction.
CIRCULAR_COLS = {
    "angle_degrees",
    "river_bearing_deg",
}


# ============================================================
# Load data
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

print(f"Loading:\n{INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

# Remove CSV index columns accidentally saved by earlier scripts.
index_cols = [
    col for col in df.columns
    if col == "Unnamed: 0" or col.startswith("Unnamed: 0.")
]
if index_cols:
    df = df.drop(columns=index_cols)
    print(f"Removed saved index columns: {index_cols}")


# ============================================================
# Prepare circular light-direction variables
# ============================================================

if "angle_degrees" in df.columns:
    angle_radians = np.deg2rad(df["angle_degrees"])
    df["light_angle_sin"] = np.sin(angle_radians)
    df["light_angle_cos"] = np.cos(angle_radians)


# ============================================================
# Identify numeric predictors
# ============================================================

excluded_cols = RESPONSE_COLS | NON_PREDICTOR_COLS | CIRCULAR_COLS

standardize_cols = [
    col
    for col in df.select_dtypes(include="number").columns
    if col not in excluded_cols
]

if not standardize_cols:
    raise ValueError("No numeric predictor columns were found to standardize.")

missing_counts = df[standardize_cols].isna().sum()
missing_counts = missing_counts[missing_counts > 0]

if not missing_counts.empty:
    raise ValueError(
        "These predictor columns contain missing values:\n"
        f"{missing_counts.to_string()}\n\n"
        "Resolve the missing values before standardization."
    )

# A constant column has a standard deviation of zero and cannot be converted
# to a z-score.
standard_deviations = df[standardize_cols].std(ddof=0)
constant_cols = standard_deviations[
    standard_deviations.isna() | (standard_deviations == 0)
].index.tolist()

if constant_cols:
    raise ValueError(
        "These predictors are constant and cannot be standardized:\n"
        f"{constant_cols}"
    )


# ============================================================
# Standardize predictors
# ============================================================

# ddof=0 matches sklearn.preprocessing.StandardScaler.
means = df[standardize_cols].mean()
standard_deviations = df[standardize_cols].std(ddof=0)

# Replace each continuous predictor with its standardized value. The original
# input CSV remains available if the unstandardized values are needed later.
df[standardize_cols] = (
    df[standardize_cols] - means
) / standard_deviations


# ============================================================
# Verify and save
# ============================================================

check = pd.DataFrame({
    "mean": df[standardize_cols].mean(),
    "standard_deviation": df[standardize_cols].std(ddof=0),
})

if not np.allclose(check["mean"], 0.0, atol=1e-10):
    raise AssertionError("At least one standardized predictor does not have mean 0.")

if not np.allclose(check["standard_deviation"], 1.0, atol=1e-10):
    raise AssertionError("At least one standardized predictor does not have SD 1.")

OUTPUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\nStandardized predictor columns:")
for col in standardize_cols:
    print(f"  {col}")

print("\nVerification (values should be mean = 0 and SD = 1):")
print(check.round(6).to_string())

unchanged_cols = [
    col for col in df.columns
    if col not in standardize_cols
]
print("\nColumns retained without standardization:")
for col in unchanged_cols:
    print(f"  {col}")

print(f"\nSaved standardized data to:\n{OUTPUT_FILE}")
