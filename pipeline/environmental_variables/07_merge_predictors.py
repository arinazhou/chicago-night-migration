from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT = Path(__file__).resolve().parents[2]

DATA = PROJECT / "data"

OUTPUT = DATA / "environmental_variables" / "output"


# ============================================================
# Input files
# ============================================================

SPECIES_FILE = (
    DATA
    / "bray-curtis_distance"
    / "bird_pivot_full_season.csv"
)

LANDCOVER_FILE = OUTPUT / "landcover_ntl_1km.csv"

RIVER_DISTANCE_FILE = OUTPUT / "river_distance.csv"

RIVER_DIRECTION_FILE = OUTPUT / "river_direction.csv"

LAKE_FILE = OUTPUT / "lake_michigan_distance.csv"

HIGHWAY_FILE = OUTPUT / "highway_predictors.csv"

LIGHT_VECTOR_FILE = OUTPUT / "light_vector.csv"

OUTPUT_FILE = OUTPUT / "environmental_predictors.csv"

# ============================================================
# Helper functions
# ============================================================

def clean_site_column(df, file_name):
    """Clean and verify the site column."""

    if "site" not in df.columns:
        raise ValueError(
            f"'site' column not found in {file_name}.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.copy()
    df["site"] = df["site"].astype(str).str.strip()

    return df

# ============================================================
# Check input paths
# ============================================================

input_files = [
    SPECIES_FILE,
    LANDCOVER_FILE,
    RIVER_DISTANCE_FILE,
    RIVER_DIRECTION_FILE,
    LAKE_FILE,
    HIGHWAY_FILE,
    LIGHT_VECTOR_FILE,
]

missing_files = [
    path for path in input_files
    if not path.exists()
]

if missing_files:
    missing_text = "\n".join(str(path) for path in missing_files)

    raise FileNotFoundError(
        "The following input files were not found:\n"
        f"{missing_text}"
    )


# ============================================================
# Load master site list
# ============================================================

print("Loading master site list...")

species = pd.read_csv(SPECIES_FILE)
species = clean_site_column(species, SPECIES_FILE)

master_sites = species[["site"]].drop_duplicates().reset_index(drop=True)


# ============================================================
# Load predictor tables
# ============================================================

print("Loading predictor tables...")

landcover = pd.read_csv(LANDCOVER_FILE)

river_distance = pd.read_csv(RIVER_DISTANCE_FILE)

river_direction = pd.read_csv(RIVER_DIRECTION_FILE)

lake = pd.read_csv(LAKE_FILE)

highway = pd.read_csv(HIGHWAY_FILE)

light_vector = pd.read_csv(LIGHT_VECTOR_FILE)

# ============================================================
# Clean site names
# ============================================================

landcover = clean_site_column(
    landcover,
    LANDCOVER_FILE
)

river_distance = clean_site_column(
    river_distance,
    RIVER_DISTANCE_FILE
)

river_direction = clean_site_column(
    river_direction,
    RIVER_DIRECTION_FILE
)

river_direction = (
    river_direction
    .drop_duplicates()
    .drop_duplicates(subset=["site"])
    .reset_index(drop=True)
)

lake = clean_site_column(
    lake,
    LAKE_FILE
)

highway = clean_site_column(
    highway,
    HIGHWAY_FILE
)

duplicate_highway_rows = highway[
    highway["site"].duplicated(keep=False)
].sort_values("site")
highway = (
    highway
    .sort_values("dist_to_highway_meter")
    .drop_duplicates(subset=["site"], keep="first")
    .reset_index(drop=True)
)

light_vector = clean_site_column(
    light_vector,
    LIGHT_VECTOR_FILE
)

# ============================================================
# Prepare light-vector predictors
# ============================================================

LIGHT_VECTOR_COLS = [
    "site",
    "light_x",
    "light_y",
    "magnitude",
    "angle_degrees",
    "weighted_light_distance_m",
    "total_radiance",
]

missing_light_cols = [
    col for col in LIGHT_VECTOR_COLS
    if col not in light_vector.columns
]

if missing_light_cols:
    raise ValueError(
        f"Missing columns in {LIGHT_VECTOR_FILE}:\n"
        f"{missing_light_cols}\n\n"
        f"Available columns:\n"
        f"{light_vector.columns.tolist()}"
    )

# Do not include latitude and longitude from light_vector.csv.
# They should already come from the land-cover/site table.
light_vector = (
    light_vector[LIGHT_VECTOR_COLS]
    .drop_duplicates(subset=["site"])
    .reset_index(drop=True)
)

# ============================================================
# Check for duplicate site rows
# ============================================================

predictor_tables = {
    "landcover": landcover,
    "river_distance": river_distance,
    "river_direction": river_direction,
    "lake": lake,
    "highway": highway,
    "light_vector": light_vector,
}

for table_name, table in predictor_tables.items():
    duplicate_sites = table.loc[
        table["site"].duplicated(keep=False),
        "site"
    ].unique()

    if len(duplicate_sites) > 0:
        raise ValueError(
            f"Duplicate sites found in {table_name}:\n"
            f"{duplicate_sites.tolist()}"
        )
        



# ============================================================
# Merge
# ============================================================

print("Merging predictor tables...")

environment = (
    master_sites

    .merge(
        landcover,
        on="site",
        how="left"
    )

    .merge(
        river_distance,
        on="site",
        how="left"
    )

    .merge(
        river_direction,
        on="site",
        how="left"
    )

    .merge(
        lake,
        on="site",
        how="left"
    )

    .merge(
        highway,
        on="site",
        how="left"
    )

    .merge(
        light_vector,
        on="site",
        how="left"
    )
)


# ============================================================
# Remove duplicate columns
# ============================================================

duplicate_cols = [
    "latitude_y",
    "longitude_y",
    "nearest_river",
    "river_bearing_deg",
]

environment = environment.drop(
    columns=[c for c in duplicate_cols if c in environment.columns]
)

environment = environment.rename(
    columns={
        "latitude_x": "latitude",
        "longitude_x": "longitude",
    }
)


# ============================================================
# Check for missing values
# ============================================================

print("\nChecking for missing values...\n")

missing = environment.isna().sum()

print(missing[missing > 0])

if missing.sum() == 0:
    print("No missing values found.")


# ============================================================
# Check all sites are retained
# ============================================================

# assert len(environment) == len(master_sites)

# assert (
#     set(environment.site)
#     == set(master_sites.site)
# )

# print(f"\nSites retained: {len(environment)}")




# ============================================================
# Save
# ============================================================

environment.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved to:\n{OUTPUT_FILE}")