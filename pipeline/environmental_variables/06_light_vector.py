"""
06_light_vector.py

Calculate nighttime-light directionality variables for each site:

    light_x
    light_y
    magnitude
    angle_degrees
    weighted_light_distance_m
    total_radiance

Output:
    light_vector.csv
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from rasterio.mask import mask
from shapely.geometry import mapping


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

NTL_RASTER = (
    DATA
    / "Nighttime_light_SDGSAT-1_processed-selected"
    / "ntl_band3_26916_radiance.tif"
)

OUT_CSV = OUTPUT / "light_vector.csv"


# =====================================================
# Parameters
# =====================================================

RADIUS_M = 1000


# =====================================================
# Light-vector function
# =====================================================

def calculate_light_vector(
    site_row,
    raster_path,
    radius_m=1000
):
    """
    Calculate brightness-weighted light direction and distance
    around one monitoring site.

    Each pixel's normalized weight is:

        weight_i = radiance_i / sum(all radiance)

    Returns
    -------
    light_x
        Brightness-weighted east-west direction.
        Positive = east, negative = west.

    light_y
        Brightness-weighted north-south direction.
        Positive = north, negative = south.

    magnitude
        Strength of directional imbalance.
        Near 0 = balanced light.
        Closer to 1 = light concentrated in one direction.

    angle_degrees
        Direction of the final light vector.
        0 = east
        90 = north
        180 / -180 = west
        -90 = south

    weighted_light_distance_m
        Brightness-weighted average distance from the site
        to surrounding light.

    total_radiance
        Sum of valid radiance values inside the 1 km buffer.
    """

    site_point = site_row.geometry
    site_buffer = site_point.buffer(radius_m)

    with rasterio.open(raster_path) as src:

        # -----------------------------------------
        # Crop raster to site's buffer
        # -----------------------------------------

        cropped, cropped_transform = mask(
            src,
            [mapping(site_buffer)],
            crop=True,
            filled=False
        )

        radiance = cropped[0]

        # -----------------------------------------
        # Pixel coordinates
        # -----------------------------------------

        rows, cols = np.indices(radiance.shape)

        xs, ys = rasterio.transform.xy(
            cropped_transform,
            rows,
            cols,
            offset="center"
        )

        xs = np.asarray(xs).reshape(radiance.shape)
        ys = np.asarray(ys).reshape(radiance.shape)

        # -----------------------------------------
        # Relative position from site
        # -----------------------------------------

        dx = xs - site_point.x
        dy = ys - site_point.y

        distance = np.sqrt(
            dx**2 + dy**2
        )

        # -----------------------------------------
        # Keep valid pixels only
        # -----------------------------------------

        valid = (
            ~np.ma.getmaskarray(radiance)
            & np.isfinite(radiance.filled(np.nan))
            & (distance <= radius_m)
            & (distance > 0)
        )

        L = radiance.filled(np.nan)[valid]

        dx_valid = dx[valid]
        dy_valid = dy[valid]
        distance_valid = distance[valid]

        # Keep nonnegative radiance
        positive = L >= 0

        L = L[positive]
        dx_valid = dx_valid[positive]
        dy_valid = dy_valid[positive]
        distance_valid = distance_valid[positive]

        # -----------------------------------------
        # Total surrounding brightness
        # -----------------------------------------

        total_radiance = L.sum()

        if total_radiance <= 0:

            return {
                "light_x": np.nan,
                "light_y": np.nan,
                "magnitude": np.nan,
                "angle_degrees": np.nan,
                "weighted_light_distance_m": np.nan,
                "total_radiance": 0
            }

        # -----------------------------------------
        # Normalize brightness
        # -----------------------------------------

        weights = L / total_radiance

        # Unit vector from site toward each pixel
        direction_x = dx_valid / distance_valid
        direction_y = dy_valid / distance_valid

        # -----------------------------------------
        # Brightness-weighted direction
        # -----------------------------------------

        light_x = np.sum(
            weights * direction_x
        )

        light_y = np.sum(
            weights * direction_y
        )

        # -----------------------------------------
        # Directional strength
        # -----------------------------------------

        magnitude = np.sqrt(
            light_x**2 + light_y**2
        )

        # -----------------------------------------
        # Direction angle
        # -----------------------------------------

        angle_degrees = np.degrees(
            np.arctan2(
                light_y,
                light_x
            )
        )

        # -----------------------------------------
        # Brightness-weighted distance
        # -----------------------------------------

        weighted_light_distance_m = np.sum(
            weights * distance_valid
        )

        return {
            "light_x": light_x,
            "light_y": light_y,
            "magnitude": magnitude,
            "angle_degrees": angle_degrees,
            "weighted_light_distance_m": weighted_light_distance_m,
            "total_radiance": total_radiance
        }


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
    .dropna(
        subset=["latitude", "longitude"]
    )
    .drop_duplicates(
        subset="site"
    )
    .reset_index(drop=True)
)

print(f"{len(sites)} unique sites loaded.")


# =====================================================
# Convert sites to same CRS as nighttime-light raster
# =====================================================

with rasterio.open(NTL_RASTER) as src:
    raster_crs = src.crs

print(f"Nighttime-light CRS: {raster_crs}")

sites = gpd.GeoDataFrame(
    sites,
    geometry=gpd.points_from_xy(
        sites.longitude,
        sites.latitude
    ),
    crs="EPSG:4326"
)

sites_utm = sites.to_crs(
    raster_crs
)


# =====================================================
# Calculate light vectors
# =====================================================

print()
print("Calculating light vectors...")

results = []

for i, (_, site_row) in enumerate(
    sites_utm.iterrows(),
    start=1
):

    site_name = site_row["site"]

    print(
        f"[{i}/{len(sites_utm)}] "
        f"{site_name}"
    )

    try:

        vector_result = calculate_light_vector(
            site_row=site_row,
            raster_path=NTL_RASTER,
            radius_m=RADIUS_M
        )

        results.append({
            "site": site_name,
            "latitude": site_row["latitude"],
            "longitude": site_row["longitude"],
            **vector_result
        })

    except Exception as e:

        print(
            f"WARNING: failed for {site_name}: {e}"
        )

        results.append({
            "site": site_name,
            "latitude": site_row["latitude"],
            "longitude": site_row["longitude"],
            "light_x": np.nan,
            "light_y": np.nan,
            "magnitude": np.nan,
            "angle_degrees": np.nan,
            "weighted_light_distance_m": np.nan,
            "total_radiance": np.nan
        })


# =====================================================
# Build output table
# =====================================================

light_vector = pd.DataFrame(
    results
)

light_vector = (
    light_vector
    .sort_values("site")
    .reset_index(drop=True)
)


# =====================================================
# Sanity checks
# =====================================================

print()
print("==============================")
print("Sanity checks")
print("==============================")

print(
    "Sites processed:",
    len(light_vector)
)

print(
    "Missing light vectors:",
    light_vector["magnitude"].isna().sum()
)

print()

print(
    light_vector[
        [
            "light_x",
            "light_y",
            "magnitude",
            "weighted_light_distance_m",
            "total_radiance"
        ]
    ].describe()
)


# Magnitude should theoretically be <= 1
too_large = light_vector[
    light_vector["magnitude"] > 1.000001
]

if len(too_large) > 0:

    print()
    print(
        "WARNING: magnitude > 1 found:"
    )

    print(
        too_large[
            [
                "site",
                "magnitude"
            ]
        ]
    )


# =====================================================
# Save
# =====================================================

light_vector.to_csv(
    OUT_CSV,
    index=False
)

print()
print("Finished!")

print(
    f"Saved to:\n{OUT_CSV}"
)