"""
01_landcover_ntl.py

Calculate:
    - 1 km land-cover proportions (NLCD 2024)
    - Mean nighttime light (SDGSAT-1)

Output:
    outputs/landcover_ntl_1km.csv
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats


# =====================================================
# Project paths
# =====================================================

# bird/
PROJECT = Path(__file__).resolve().parents[2]

# bird/data/
DATA = PROJECT / "data"

# bird/data/environmental_variables/output/
OUTPUT = DATA / "environmental_variables" / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

# =====================================================
# Input files
# =====================================================

SITE_FILE = (
    DATA
    / "bray-curtis_distance"
    / "lat&lon.xlsx"
)

NLCD_RASTER = (
    DATA
    / "landcover_NLCD_tiff"
    / "Annual_NLCD_LndCov_2024_CU_C1V1_mgucdm90juetpn.tiff"
)

NTL_RASTER = (
    DATA
    / "Nighttime_light_SDGSAT-1_processed-selected"
    / "ntl_band3_26916_radiance.tif"
)

# =====================================================
# Output files
# =====================================================

OUT_CSV = OUTPUT / "landcover_ntl_1km.csv"

# =====================================================
# Parameters
# =====================================================

BUFFER_RADIUS = 1000  # meters


# =====================================================
# NLCD legend
# =====================================================

NLCD_CLASSES = {
    11: "Open Water",
    12: "Perennial Ice/Snow",

    21: "Developed, Open Space",
    22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity",
    24: "Developed, High Intensity",

    31: "Barren Land",

    41: "Deciduous Forest",
    42: "Evergreen Forest",
    43: "Mixed Forest",

    51: "Dwarf Scrub",
    52: "Shrub/Scrub",

    71: "Grassland/Herbaceous",
    72: "Sedge/Herbaceous",

    73: "Lichens",
    74: "Moss",

    81: "Pasture/Hay",
    82: "Cultivated Crops",

    90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands",

    0: "NoData",
}


LC_ORDER = [

    "Open Water",

    "Developed, Open Space",
    "Developed, Low Intensity",
    "Developed, Medium Intensity",
    "Developed, High Intensity",

    "Deciduous Forest",
    "Evergreen Forest",
    "Mixed Forest",

    "Shrub/Scrub",

    "Grassland/Herbaceous",

    "Pasture/Hay",
    "Cultivated Crops",

    "Woody Wetlands",
    "Emergent Herbaceous Wetlands",

    "Barren Land",
]


# =====================================================
# Load monitoring sites
# =====================================================

print("Loading monitoring sites...")

sites = pd.read_excel(SITE_FILE)

sites = sites.rename(
    columns={
        "Location": "site",
        "Latitude": "latitude",
        "Longtitude": "longitude",
    }
)

sites = (
    sites
    .dropna(subset=["latitude", "longitude"])
    .drop_duplicates(subset="site")
    .reset_index(drop=True)
)


sites = gpd.GeoDataFrame(

    sites,

    geometry=gpd.points_from_xy(
        sites.longitude,
        sites.latitude
    ),

    crs="EPSG:4326",
)


# =====================================================
# Calculate land-cover + NTL
# =====================================================

print("Opening rasters...")

with rasterio.open(NLCD_RASTER) as nlcd, rasterio.open(NTL_RASTER) as ntl:

    sites_proj = sites.to_crs(nlcd.crs)

    # --------------------------------------------
    # Land-cover class directly beneath each site
    # --------------------------------------------

    coords = list(
        zip(
            sites_proj.geometry.x,
            sites_proj.geometry.y,
        )
    )

    values = []

    for v in nlcd.sample(coords):

        if v is None:
            values.append(0)

        elif len(v) == 0:
            values.append(0)

        elif np.isnan(v[0]):
            values.append(0)

        else:
            values.append(int(v[0]))

    sites_proj["land cover class code"] = values

    sites_proj["landcover_class_point"] = (
        sites_proj["land cover class code"]
        .map(NLCD_CLASSES)
    )

    # --------------------------------------------
    # 1 km buffer
    # --------------------------------------------

    print("Calculating 1 km land-cover proportions...")

    buffers = sites_proj.copy()

    buffers.geometry = buffers.buffer(BUFFER_RADIUS)

    stats = zonal_stats(
        buffers,
        NLCD_RASTER,
        categorical=True,
        nodata=nlcd.nodata,
        all_touched=False,
    )

    stats = (
        pd.DataFrame(stats)
        .fillna(0)
        .astype(int)
    )

    stats = stats.rename(columns=NLCD_CLASSES)

    totals = stats.sum(axis=1).replace(0, np.nan)

    props = stats.div(totals, axis=0)

    for col in LC_ORDER:

        if col not in props.columns:
            props[col] = 0.0

    props = props[LC_ORDER]

    # --------------------------------------------
    # Mean nighttime light
    # --------------------------------------------

    print("Calculating mean nighttime light...")

    buffers_ntl = buffers.to_crs(ntl.crs)

    ntl_stats = zonal_stats(

        buffers_ntl,

        NTL_RASTER,

        stats=["mean"],

        nodata=ntl.nodata,

        all_touched=False,

    )

    mean_radiance = [
        x["mean"]
        for x in ntl_stats
    ]

    # --------------------------------------------
    # Assemble output
    # --------------------------------------------

    base = buffers.drop(columns="geometry").reset_index(drop=True)

    out = pd.concat(
        [
            base,
            props.reset_index(drop=True),
        ],
        axis=1,
    )

    out["mean_radiance"] = mean_radiance

    keep = [

    "site",

    "latitude",
    "longitude",

    "land cover class code",
    "landcover_class_point",

    ] + LC_ORDER + [

        "mean_radiance"

    ]

    keep = [
        c
        for c in keep
        if c in out.columns
    ]

    out = out[keep]


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