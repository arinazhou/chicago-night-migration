"""
05_highway.py

Calculate:
    - Distance to nearest highway
    - Highway richness (number of highway segments within 1 km)

Output:
    highway_predictors.csv
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

# =====================================================
# Project paths
# =====================================================

PROJECT = Path(__file__).resolve().parents[2]

DATA = PROJECT / "data"

OUTPUT = DATA / "environmental_variables" / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

SITE_FILE = (
    DATA
    / "bray-curtis_distance"
    / "lat&lon.xlsx"
)

HIGHWAY_FILE = (
    DATA
    / "highways"
    / "IL_highways.shp"
)

OUT_CSV = OUTPUT / "highway_predictors.csv"

BUFFER_RADIUS = 1000  # meters


# =====================================================
# Load monitoring sites
# =====================================================

print("Loading monitoring sites...")

sites = (
    pd.read_excel(SITE_FILE)
    .rename(
        columns={
            "Location": "site",
            "Latitude": "latitude",
            "Longtitude": "longitude",
        }
    )
    .dropna(subset=["latitude", "longitude"])
    .drop_duplicates(subset="site")
    .reset_index(drop=True)
)

sites = gpd.GeoDataFrame(
    sites,
    geometry=gpd.points_from_xy(
        sites.longitude,
        sites.latitude,
    ),
    crs="EPSG:4326",
)


# =====================================================
# Load highways
# =====================================================

print("Loading highways...")

roads = gpd.read_file(HIGHWAY_FILE)

sites = sites.to_crs("EPSG:26916")
roads = roads.to_crs("EPSG:26916")


# =====================================================
# Distance to nearest highway
# =====================================================

print("Calculating nearest highway distance...")

nearest = gpd.sjoin_nearest(
    sites,
    roads,
    how="left",
    distance_col="dist_to_highway_meter",
)


# =====================================================
# Highway richness
# =====================================================

print("Calculating highway richness...")

buffers = sites.copy()
buffers.geometry = buffers.buffer(BUFFER_RADIUS)

joined = gpd.sjoin(
    buffers,
    roads,
    how="left",
    predicate="intersects",
)

richness = (
    joined.groupby("site")
    .size()
    .rename("highway_richness")
)

out = (
    nearest[
        [
            "site",
            "dist_to_highway_meter",
        ]
    ]
    .merge(
        richness,
        on="site",
        how="left",
    )
)

out["highway_richness"] = (
    out["highway_richness"]
    .fillna(0)
    .astype(int)
)

out = out.sort_values("site").reset_index(drop=True)


# =====================================================
# Save
# =====================================================

out.to_csv(
    OUT_CSV,
    index=False,
)

print()
print("Finished!")
print(f"Sites processed: {len(out)}")
print(f"Saved to:\n{OUT_CSV}")