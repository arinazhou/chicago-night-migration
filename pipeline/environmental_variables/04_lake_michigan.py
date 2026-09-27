"""
04_lake_michigan.py

Calculate the distance from each monitoring site to the shoreline
of Lake Michigan.

Output:
    lake_michigan_distance.csv
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

LAKES_SHP = (
    DATA
    / "OSP_illinois_waterways"
    / "ne_10m_lakes.shp"
)

OUT_CSV = OUTPUT / "lake_michigan_distance.csv"


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

print(f"{len(sites)} sites loaded.")


# =====================================================
# Load Lake Michigan
# =====================================================

print("Loading lakes...")

lakes = gpd.read_file(LAKES_SHP)

lake_michigan = lakes[
    lakes["name"].str.contains(
        "Michigan",
        case=False,
        na=False,
    )
].copy()

if len(lake_michigan) != 1:
    raise ValueError(
        f"Expected one Lake Michigan polygon, found {len(lake_michigan)}."
    )

print("Lake Michigan found.")


# =====================================================
# Project to meters
# =====================================================

sites = sites.to_crs("EPSG:26916")
lake_michigan = lake_michigan.to_crs("EPSG:26916")

shoreline = lake_michigan.boundary


# =====================================================
# Distance to shoreline
# =====================================================

print("Calculating shoreline distance...")

sites["dist_LakeMichigan"] = sites.geometry.apply(
    lambda point: shoreline.distance(point).min()
)


# =====================================================
# Save
# =====================================================

out = sites[
    [
        "site",
        "dist_LakeMichigan",
    ]
].copy()

out = out.sort_values("site").reset_index(drop=True)

out.to_csv(
    OUT_CSV,
    index=False,
)

print()
print("Finished!")
print(f"Sites processed: {len(out)}")
print(f"Saved to:\n{OUT_CSV}")