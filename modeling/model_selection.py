"""
model_selection.py

Exhaustive OLS model search for the dominant community axis (NMDS1).

Every 1-, 2- and 3-predictor combination of the candidate predictors is fit
(92 models) and ranked by AIC. Predictors come from
environmental_predictors_standardized.csv, so coefficients are comparable.

Input:
    analysis_fullseason.csv   (site x [NMDS1, NMDS2, standardized predictors])

Output:
    model_selection_results.csv
"""

from itertools import combinations
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent

INPUT_FILE = HERE / "analysis_fullseason.csv"
OUTPUT_FILE = HERE / "model_selection_results.csv"

RESPONSE = "NMDS1"
MAX_PREDICTORS = 3


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_FILE)


# ============================================================
# Composite land-cover predictors
# ============================================================

# Development classes weighted by intensity (open space = 1 ... high = 4)
df["urbanization_index"] = (
    df["Developed, Open Space"] * 1
    + df["Developed, Low Intensity"] * 2
    + df["Developed, Medium Intensity"] * 3
    + df["Developed, High Intensity"] * 4
)

df["forest"] = (
    df["Deciduous Forest"]
    + df["Evergreen Forest"]
    + df["Mixed Forest"]
)

df["nonforest_natural_score"] = (
    df["Woody Wetlands"]
    + df["Emergent Herbaceous Wetlands"]
    + df["Grassland/Herbaceous"]
    + df["Shrub/Scrub"]
    + df["Cultivated Crops"]
    + df["Pasture/Hay"]
)

CANDIDATE_PREDICTORS = [
    "urbanization_index",
    "forest",
    "nonforest_natural_score",
    "distance_to_river_m",
    "dist_LakeMichigan",
    "mean_radiance",
    "latitude",
    "longitude",
]


# ============================================================
# Fit every combination
# ============================================================

results = []

for k in range(1, MAX_PREDICTORS + 1):
    for predictors in combinations(CANDIDATE_PREDICTORS, k):
        X = sm.add_constant(df[list(predictors)])
        model = sm.OLS(df[RESPONSE], X).fit()

        results.append({
            "predictors": ", ".join(predictors),
            "n_predictors": k,
            "AIC": model.aic,
            "adj_R2": model.rsquared_adj,
        })

results = pd.DataFrame(results).sort_values("AIC").reset_index(drop=True)
results["delta_AIC"] = results["AIC"] - results["AIC"].min()


# ============================================================
# Report best model
# ============================================================

best = results.loc[0, "predictors"].split(", ")
best_model = sm.OLS(df[RESPONSE], sm.add_constant(df[best])).fit()

print(f"Models fit: {len(results)}")
print(results.head(10).to_string(index=False))
print()
print(best_model.summary())

results.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to:\n{OUTPUT_FILE}")
