"""
03_river_direction.py

Calculate the local orientation of the nearest major river.

Output:
    river_direction.csv
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import substring


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

RIVER_SHP = (
    DATA
    / "OSP_illinois_waterways"
    / "gis_osm_waterways_free_1.shp"
)

OUT_CSV = OUTPUT / "river_direction.csv"


# =====================================================
# Rivers
# =====================================================

MAJOR_RIVERS = [
    "Fox River",
    "Chicago River",
    "Illinois River",
    "Des Plaines River",
]


# =====================================================
# Load sites
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
# Load rivers
# =====================================================

rivers = gpd.read_file(RIVER_SHP)

rivers = rivers[
    rivers["name"].isin(MAJOR_RIVERS)
].copy()

sites = sites.to_crs("EPSG:26916")
rivers = rivers.to_crs("EPSG:26916")


# =====================================================
# Find nearest river
# =====================================================

joined = gpd.sjoin_nearest(
    sites,
    rivers,
    how="left",
).reset_index(drop=True)

joined["river_geom"] = joined["index_right"].map(
    rivers.geometry
)


# =====================================================
# Helper functions
# =====================================================

def snap_point(line, point):
    return line.interpolate(line.project(point))


def nearest_segment(line, point, radius=500):

    if line.geom_type == "MultiLineString":
        line = min(
            line.geoms,
            key=lambda g: g.distance(point)
        )

    length = line.length

    d = line.project(point)

    seg = substring(
        line,
        max(0, d - radius),
        min(length, d + radius),
    )

    if seg.length == 0:
        seg = substring(
            line,
            max(0, d - 5),
            min(length, d + 5),
        )

    return seg


def bearing(segment):

    x0, y0 = segment.coords[0]
    x1, y1 = segment.coords[-1]

    dx = x1 - x0
    dy = y1 - y0

    angle = np.degrees(
        np.arctan2(dx, dy)
    )

    return (angle + 360) % 360


# =====================================================
# Calculate river direction
# =====================================================

print("Calculating river orientation...")

angles = []

for _, row in joined.iterrows():

    line = row["river_geom"]
    pt = row.geometry

    snapped = snap_point(line, pt)

    seg = nearest_segment(
        line,
        snapped,
        radius=500,
    )

    angles.append(
        bearing(seg)
    )

joined["river_bearing_deg"] = angles

joined["river_bearing_sin"] = np.sin(
    np.deg2rad(joined["river_bearing_deg"])
)

joined["river_bearing_cos"] = np.cos(
    np.deg2rad(joined["river_bearing_deg"])
)


# =====================================================
# Save
# =====================================================

out = joined[
    [
        "site",
        "river_bearing_deg",
        "river_bearing_sin",
        "river_bearing_cos",
    ]
].copy()

out = out.sort_values("site").reset_index(drop=True)

out.to_csv(
    OUT_CSV,
    index=False,
)

print()
print("Finished!")
print(f"Saved to:\n{OUT_CSV}")