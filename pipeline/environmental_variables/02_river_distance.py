"""
02_river_distance.py

Calculate the distance from each monitoring site to the nearest
major river.

Output:
    river_distance.csv
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
# OUTPUT.mkdir(parents=True, exist_ok=True)

SITE_FILE = (
    DATA
    / "bray-curtis_distance"
    / "lat&lon.xlsx"
)

RIVER_SHP = (
    DATA
    / "OSP_illinois_waterways"
    / "gis_osm_waterways_free_1.shp"
)

OUT_CSV = OUTPUT / "river_distance.csv"


# =====================================================
# Rivers to include
# =====================================================

MAJOR_RIVERS = [
    "Fox River",
    "Chicago River",
    "Illinois River",
    "Des Plaines River",
]


# =====================================================
# Load monitoring sites
# =====================================================

print("Loading monitoring sites...")

sites = pd.read_excel(SITE_FILE)

sites = (
    sites.rename(
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

print(f"{len(sites)} sites loaded.")


# =====================================================
# Load rivers
# =====================================================

print("Loading rivers...")

rivers = gpd.read_file(RIVER_SHP)

rivers = rivers[
    rivers["name"].isin(MAJOR_RIVERS)
].copy()

print(f"{len(rivers)} river segments retained.")


# =====================================================
# Project to meters
# =====================================================

sites = sites.to_crs("EPSG:26916")
rivers = rivers.to_crs("EPSG:26916")


# =====================================================
# Nearest river
# =====================================================

print("Finding nearest river...")

nearest = gpd.sjoin_nearest(
    sites,
    rivers,
    how="left",
    distance_col="distance_to_river_m",
)

out = nearest[
    [
        "site",
        "latitude",
        "longitude",
        "name",
        "distance_to_river_m",
    ]
].copy()

out = out.rename(
    columns={
        "name": "nearest_river"
    }
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