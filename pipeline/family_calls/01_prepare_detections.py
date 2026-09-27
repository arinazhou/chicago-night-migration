"""
prepare_family_call_data_safe.py

SAFE preprocessing for Van Doren Lab family-call analysis.

What this script does:
- Reads Fall 2025 and Spring 2026 raw detector exports.
- NEVER overwrites the raw files.
- NEVER silently deletes rows.
- Preserves the source file and original row number for every detection.
- Adds review flags instead of automatically removing uncertain rows.
- Applies the nocturnal rule as a FLAG, not destructive filtering.
- Exports separate Fall, Spring, and Full-season nocturnal datasets.
- Exports review tables for taxonomy, duplicates, missing names, etc.

IMPORTANT:
This is still the SAFE preprocessing stage.
It does NOT automatically assign Family1.
It does NOT automatically delete questionable taxa.
"""

from pathlib import Path
import hashlib
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

# This script should be located at:
# bird/data/family_calls_data/prepare_family_call_data_safe.py

FAMILY_DIR = Path(__file__).resolve().parent
DATA = FAMILY_DIR.parent

RAW_DIR = DATA / "bray-curtis_distance" / "fullseason_data"

OUTPUT_DIR = FAMILY_DIR / "family_call_preprocessing_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILES = [
    RAW_DIR
    / "export_fall2025_163-181-184-americas-chicago_nautical-dawn-dusk_confidence80-10.csv",

    RAW_DIR
    / "export_spring2026_314-americas-chicago_nautical-dawn-dusk_confidence80-10.csv",
]


# ============================================================
# 2. NIGHTTIME SETTINGS
# ============================================================

# Previous definition:
# sunset + 30 min  ->  sunrise - 30 min

MIN_FROM_SUNSET = 30
MIN_BEFORE_SUNRISE = 30


# ============================================================
# 3. YOUR EXISTING COMMON NAME -> SCIENTIFIC NAME MAP
# ============================================================

name_map = {
    "Ovenbird": "Seiurus aurocapilla",
    "Killdeer": "Charadrius vociferus",
    "Swainson's Thrush": "Catharus swainsoni",
    "Rose-breasted Grosbeak": "Pheucticus ludovicianus",
    "Savannah Sparrow": "Passerculus guttatus",
    "White-crowned Sparrow": "Zonotrichia leucophrys",
    "American Robin": "Turdus confinis",
    "Palm Warbler": "Setophaga palmarum",
    "White-throated Sparrow": "Zonotrichia albicollis",
    "American Redstart": "Setophaga ruticilla",
    "Indigo Bunting": "Passerina cyanea",
    "American Pipit": "Anthus rubescens",
    "Cape May Warbler": "Setophaga tigrina",
    "Red-breasted Nuthatch": "Sitta canadensis",
    "Common Yellowthroat": "Geothlypis trichas",
    "Chipping Sparrow": "Spizella passerina",
    "Green Heron": "Ardeola rufiventris",
    "American Golden-Plover": "Pluvialis dominica",
    "Black-and-white Warbler": "Mniotilta varia",
    "Lincoln's Sparrow": "Melospiza lincolnii",
    "Magnolia Warbler": "Setophaga magnolia",
    "Dickcissel": "Spiza americana",
    "Northern Waterthrush": "Parkesia noveboracensis",
    "Northern Parula": "Setophaga americana",
    "Grasshopper Sparrow": "Ammodramus savannarum",
    "Bobolink": "Dolichonyx oryzivorus",
    "Black-crowned Night Heron": "Nycticorax nycticorax",
    "Wood Thrush": "Hylocichla mustelina",
    "Chestnut-sided Warbler": "Setophaga pensylvanica",
    "Gray-cheeked Thrush": "Catharus minimus",
    "Yellow-rumped Warbler": "Setophaga coronata",
    "Great Blue Heron": "Ardea herodias",
    "Common Nighthawk": "Chordeiles minor",
    "Veery": "Catharus fuscescens",
    "Mourning Warbler": "Geothlypis philadelphia",
    "Scarlet Tanager": "Piranga olivacea",
    "Hooded Warbler": "Setophaga citrina",
    "Dark-eyed Junco": "Junco hyemalis",
    "Black-billed Cuckoo": "Coccyzus erythropthalmus",
    "Wilson's Snipe": "Gallinago delicata",
    "Golden-crowned Kinglet": "Regulus satrapa",
    "Least Sandpiper": "Calidris minutilla",
    "American Bittern": "Botaurus lentiginosus",
    "Solitary Sandpiper": "Tringa solitaria",
    "Semipalmated Plover": "Charadrius vociferus",
    "American Tree Sparrow": "Spizelloides arborea",
    "Least Bittern": "Ixobrychus exilis",
    "Spotted Sandpiper": "Actitis macularius",
    "Black-bellied Plover": "Pluvialis squatarola",
    "LeConte's Sparrow": "Ammospiza leconteii",
    "Caspian Tern": "Hydroprogne caspia",
    "Baird's Sandpiper": "Calidris bairdii",
    "Yellow-crowned Night Heron": "Nyctanassa violacea",
    "Hermit Thrush": "Catharus guttatus",
    "Upland Sandpiper": "Bartramia longicauda",
    "Red Knot": "Calidris canutus",
    "Wilson's Warbler": "Cardellina pusilla",
    "Greater Yellowlegs": "Tringa melanoleuca",
    "Dunlin": "Calidris alpina",
    "Canada Warbler": "Cardellina canadensis",
    "Cedar Waxwing": "Bombycilla cedrorum",
    "Ruddy Turnstone": "Arenaria interpres",
    "Canada Goose": "Branta canadensis",
    "Vesper Sparrow": "Pooecetes gramineus",
    "Lesser Yellowlegs": "Tringa flavipes",
    "Long-billed Curlew": "Numenius americanus",
    "Piping Plover": "Charadrius melodus",
    "Nashville Warbler": "Leiothlypis ruficapilla",
}


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def file_sha256(path):
    """Checksum so we can verify the raw file itself was not changed."""
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)

    return sha.hexdigest()


def safe_read_csv(path):
    """
    Read one raw CSV without modifying it.
    Adds source-file and source-row bookkeeping columns.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"\nCould not find raw file:\n{path}\n"
        )

    df = pd.read_csv(path, low_memory=False)

    # Track exact origin of every row.
    df.insert(0, "_source_file", path.name)

    # +2 because CSV line 1 is the header.
    df.insert(1, "_source_row", range(2, len(df) + 2))

    return df


# ============================================================
# 5. CHECK PATHS
# ============================================================

print_section("PATH CHECK")

print("DATA:")
print(DATA)

print("\nRAW_DIR:")
print(RAW_DIR)

print("\nOUTPUT_DIR:")
print(OUTPUT_DIR)

print("\nRaw files:")

for path in RAW_FILES:
    print(f"{path.name}: {path.exists()}")

    if not path.exists():
        raise FileNotFoundError(
            f"Raw file does not exist:\n{path}"
        )


# ============================================================
# 6. LOAD RAW FILES
# ============================================================

print_section("LOAD RAW FILES")

manifest_rows = []
dataframes = []

for path in RAW_FILES:

    df = safe_read_csv(path)

    checksum = file_sha256(path)

    manifest_rows.append({
        "source_file": path.name,
        "absolute_path": str(path.resolve()),
        "n_rows": len(df),
        "n_columns_original": len(df.columns) - 2,
        "sha256": checksum,
    })

    dataframes.append(df)

    print(f"\n{path.name}")
    print(f"Rows:   {len(df):,}")
    print(f"SHA256: {checksum}")


manifest = pd.DataFrame(manifest_rows)

manifest.to_csv(
    OUTPUT_DIR / "raw_file_manifest.csv",
    index=False
)


# ============================================================
# 7. COMBINE WITHOUT REMOVING ANY ROWS
# ============================================================

print_section("COMBINE RAW FILES")

original_row_total = sum(len(df) for df in dataframes)

all_data = pd.concat(
    dataframes,
    ignore_index=True,
    sort=False
)

if len(all_data) != original_row_total:
    raise RuntimeError(
        "Row count changed unexpectedly during concatenation."
    )

all_data.insert(
    0,
    "_combined_row_id",
    range(1, len(all_data) + 1)
)

print(f"Total raw rows: {original_row_total:,}")
print(f"Combined rows:  {len(all_data):,}")


# ============================================================
# 8. REQUIRED COLUMN CHECK
# ============================================================

print_section("COLUMN CHECK")

required_columns = [
    "site",
    "timestamp",
    "commonName",
    "taxonCode",
    "sunriseOffsetMin",
    "sunsetOffsetMin",
]

missing_required_columns = [
    col for col in required_columns
    if col not in all_data.columns
]

if missing_required_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_required_columns)
    )

print("Required columns found.")


# ============================================================
# 9. BASIC DATA FLAGS
# ============================================================

print_section("BASIC DATA FLAGS")

# Keep original timestamp unchanged.
# Create a separate parsed version.
all_data["_timestamp_parsed"] = pd.to_datetime(
    all_data["timestamp"],
    errors="coerce",
    utc=True
)

all_data["_flag_missing_site"] = (
    all_data["site"].isna()
    | (all_data["site"].astype(str).str.strip() == "")
)

all_data["_flag_invalid_timestamp"] = (
    all_data["_timestamp_parsed"].isna()
)

all_data["_flag_missing_common_name"] = (
    all_data["commonName"].isna()
    | (
        all_data["commonName"]
        .fillna("")
        .astype(str)
        .str.strip()
        == ""
    )
)

all_data["_flag_missing_taxon_code"] = (
    all_data["taxonCode"].isna()
    | (
        all_data["taxonCode"]
        .fillna("")
        .astype(str)
        .str.strip()
        == ""
    )
)


# ============================================================
# 10. TAXONOMY REVIEW FLAGS
# ============================================================

print_section("TAXONOMY REVIEW FLAGS")

expected_names = set(name_map.keys())

all_data["_flag_name_not_in_name_map"] = (
    ~all_data["_flag_missing_common_name"]
    & ~all_data["commonName"].isin(expected_names)
)

# Only use YOUR map.
# Do not guess unknown scientific names.
all_data["_scientific_name_from_your_map"] = (
    all_data["commonName"].map(name_map)
)

# Family-level codes often end in -idae / idae.
# This is only a review flag.
all_data["_flag_possible_family_level_taxon"] = (
    all_data["taxonCode"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.endswith("idae")
)

# Flag likely grouped/non-species taxon codes.
# DO NOT remove them automatically.
all_data["_flag_possible_group_taxon"] = (
    all_data["_flag_missing_common_name"]
    | all_data["_flag_possible_family_level_taxon"]
    | (
        all_data["taxonCode"]
        .fillna("")
        .astype(str)
        .str.contains("-", regex=False)
    )
)


# ============================================================
# 11. NIGHTTIME FLAG
# ============================================================

print_section("NOCTURNAL WINDOW")

sunrise = pd.to_numeric(
    all_data["sunriseOffsetMin"],
    errors="coerce"
)

sunset = pd.to_numeric(
    all_data["sunsetOffsetMin"],
    errors="coerce"
)

all_data["_flag_missing_sun_offsets"] = (
    sunrise.isna()
    | sunset.isna()
)

# Old nocturnal definition:
#
# >= 30 minutes after sunset
# OR
# >= 30 minutes before sunrise

all_data["_passes_nocturnal_window"] = (
    (sunset >= MIN_FROM_SUNSET)
    | (sunrise <= -MIN_BEFORE_SUNRISE)
)

# Missing offsets are not considered confidently nocturnal.
all_data.loc[
    all_data["_flag_missing_sun_offsets"],
    "_passes_nocturnal_window"
] = False

all_data["_flag_outside_nocturnal_window"] = (
    ~all_data["_passes_nocturnal_window"]
    & ~all_data["_flag_missing_sun_offsets"]
)


# ============================================================
# 12. DUPLICATE FLAGS
# ============================================================

print_section("DUPLICATE CHECK")

if "annotationId" in all_data.columns:

    all_data["_flag_duplicate_annotation_id"] = (
        all_data["annotationId"].notna()
        & all_data.duplicated(
            subset=["annotationId"],
            keep=False
        )
    )

else:
    all_data["_flag_duplicate_annotation_id"] = False


possible_detection_key = [
    col
    for col in [
        "site",
        "timestamp",
        "taxonCode",
        "startSec",
        "endSec",
        "recordingId",
    ]
    if col in all_data.columns
]

if len(possible_detection_key) > 0:

    all_data["_flag_duplicate_detection_key"] = (
        all_data.duplicated(
            subset=possible_detection_key,
            keep=False
        )
    )

else:
    all_data["_flag_duplicate_detection_key"] = False


# ============================================================
# 13. MASTER MANUAL REVIEW FLAG
# ============================================================

review_flag_columns = [
    "_flag_missing_site",
    "_flag_invalid_timestamp",
    "_flag_missing_common_name",
    "_flag_missing_taxon_code",
    "_flag_name_not_in_name_map",
    "_flag_possible_family_level_taxon",
    "_flag_possible_group_taxon",
    "_flag_missing_sun_offsets",
    "_flag_duplicate_annotation_id",
    "_flag_duplicate_detection_key",
]

all_data["_needs_manual_review"] = (
    all_data[review_flag_columns]
    .any(axis=1)
)


# ============================================================
# 14. SAVE COMPLETE MASTER DATASET
# ============================================================

print_section("SAVE MASTER DATASET")

master_path = (
    OUTPUT_DIR
    / "all_detections_with_flags.csv"
)

all_data.to_csv(
    master_path,
    index=False
)

# Re-read only one column to verify the number of saved rows.
saved_check = pd.read_csv(
    master_path,
    usecols=["_combined_row_id"]
)

if len(saved_check) != original_row_total:
    raise RuntimeError(
        "Saved master dataset has a different number of rows "
        "than the raw files."
    )

print(f"Master rows saved: {len(saved_check):,}")
print("Row-count safety check: PASSED")


# ============================================================
# 15. MANUAL REVIEW EXPORTS
# ============================================================

print_section("SAVE REVIEW FILES")


# ------------------------------------------------------------
# Rows needing manual review
# ------------------------------------------------------------

review_rows = all_data.loc[
    all_data["_needs_manual_review"]
].copy()

review_rows.to_csv(
    OUTPUT_DIR
    / "rows_requiring_manual_review.csv",
    index=False
)


# ------------------------------------------------------------
# Unique taxonomy summary
# ------------------------------------------------------------

taxonomy_summary = (
    all_data
    .groupby(
        ["commonName", "taxonCode"],
        dropna=False
    )
    .agg(
        n_rows=("_combined_row_id", "size"),
        n_sites=("site", "nunique"),
        missing_common_name=(
            "_flag_missing_common_name",
            "max"
        ),
        name_not_in_name_map=(
            "_flag_name_not_in_name_map",
            "max"
        ),
        possible_family_level_taxon=(
            "_flag_possible_family_level_taxon",
            "max"
        ),
        possible_group_taxon=(
            "_flag_possible_group_taxon",
            "max"
        ),
    )
    .reset_index()
)

taxonomy_summary[
    "scientific_name_from_your_map"
] = taxonomy_summary["commonName"].map(name_map)

taxonomy_summary = taxonomy_summary.sort_values(
    [
        "name_not_in_name_map",
        "missing_common_name",
        "n_rows"
    ],
    ascending=[
        False,
        False,
        False
    ]
)

taxonomy_summary.to_csv(
    OUTPUT_DIR
    / "taxonomy_review_summary.csv",
    index=False
)


# ------------------------------------------------------------
# Names absent from your map
# ------------------------------------------------------------

names_not_in_map = (
    all_data.loc[
        all_data["_flag_name_not_in_name_map"],
        ["commonName", "taxonCode"]
    ]
    .groupby(
        ["commonName", "taxonCode"],
        dropna=False
    )
    .size()
    .reset_index(name="n_rows")
    .sort_values(
        "n_rows",
        ascending=False
    )
)

names_not_in_map.to_csv(
    OUTPUT_DIR
    / "names_not_in_name_map.csv",
    index=False
)


# ------------------------------------------------------------
# taxonCode when commonName is missing
# ------------------------------------------------------------

missing_name_taxa = (
    all_data.loc[
        all_data["_flag_missing_common_name"],
        ["taxonCode"]
    ]
    .groupby(
        "taxonCode",
        dropna=False
    )
    .size()
    .reset_index(name="n_rows")
    .sort_values(
        "n_rows",
        ascending=False
    )
)

missing_name_taxa.to_csv(
    OUTPUT_DIR
    / "missing_common_name_taxa.csv",
    index=False
)


# ------------------------------------------------------------
# Possible duplicate rows
# ------------------------------------------------------------

duplicate_rows = all_data.loc[
    all_data["_flag_duplicate_annotation_id"]
    | all_data["_flag_duplicate_detection_key"]
].copy()

duplicate_rows.to_csv(
    OUTPUT_DIR
    / "possible_duplicate_rows.csv",
    index=False
)


# ------------------------------------------------------------
# Outside nighttime window
# ------------------------------------------------------------

outside_night = all_data.loc[
    all_data["_flag_outside_nocturnal_window"]
].copy()

outside_night.to_csv(
    OUTPUT_DIR
    / "outside_nocturnal_window.csv",
    index=False
)


# ------------------------------------------------------------
# Missing sunrise/sunset offsets
# ------------------------------------------------------------

missing_sun = all_data.loc[
    all_data["_flag_missing_sun_offsets"]
].copy()

missing_sun.to_csv(
    OUTPUT_DIR
    / "missing_sun_offset_rows.csv",
    index=False
)


# ============================================================
# 16. CREATE NOCTURNAL BASE DATASET
# ============================================================

print_section("CREATE NOCTURNAL DATASETS")

# This creates a derived candidate dataset.
# The master all_detections_with_flags.csv still contains EVERYTHING.

nocturnal_candidates = all_data.loc[
    all_data["_passes_nocturnal_window"]
].copy()

nocturnal_candidates.to_csv(
    OUTPUT_DIR
    / "nocturnal_candidates.csv",
    index=False
)


# ============================================================
# 17. SPLIT FALL / SPRING / FULL SEASON
# ============================================================

# ------------------------------------------------------------
# FALL 2025
# ------------------------------------------------------------

fall_data = nocturnal_candidates.loc[
    nocturnal_candidates["_source_file"]
    .str.contains(
        "fall2025",
        case=False,
        na=False
    )
].copy()

fall_path = (
    OUTPUT_DIR
    / "fall2025_nocturnal_detections.csv"
)

fall_data.to_csv(
    fall_path,
    index=False
)


# ------------------------------------------------------------
# SPRING 2026
# ------------------------------------------------------------

spring_data = nocturnal_candidates.loc[
    nocturnal_candidates["_source_file"]
    .str.contains(
        "spring2026",
        case=False,
        na=False
    )
].copy()

spring_path = (
    OUTPUT_DIR
    / "spring2026_nocturnal_detections.csv"
)

spring_data.to_csv(
    spring_path,
    index=False
)


# ------------------------------------------------------------
# FULL SEASON
# ------------------------------------------------------------

# Full season = Fall + Spring detections.
# Do NOT average them.

fullseason_data = pd.concat(
    [
        fall_data,
        spring_data
    ],
    ignore_index=True
)

fullseason_path = (
    OUTPUT_DIR
    / "fullseason_nocturnal_detections.csv"
)

fullseason_data.to_csv(
    fullseason_path,
    index=False
)


# ============================================================
# 18. SEASON SAFETY CHECKS
# ============================================================

print_section("SEASON SAFETY CHECKS")

expected_fullseason_rows = (
    len(fall_data)
    + len(spring_data)
)

if len(fullseason_data) != expected_fullseason_rows:
    raise RuntimeError(
        "Full-season row count is not equal to "
        "Fall + Spring row counts."
    )

if len(nocturnal_candidates) != expected_fullseason_rows:
    print(
        "WARNING: nocturnal_candidates contains rows "
        "that were not identified as fall2025 or spring2026."
    )

print(f"Fall 2025 rows:       {len(fall_data):,}")
print(f"Spring 2026 rows:     {len(spring_data):,}")
print(f"Full-season rows:     {len(fullseason_data):,}")
print(f"Nocturnal candidates: {len(nocturnal_candidates):,}")

print("\nFall + Spring = Full season check: PASSED")


# ============================================================
# 19. PRINT REVIEW SUMMARY
# ============================================================

print_section("REVIEW SUMMARY")

print(
    f"Total raw detections: "
    f"{len(all_data):,}"
)

print(
    f"Rows requiring manual review: "
    f"{len(review_rows):,}"
)

print(
    f"Rows with missing commonName: "
    f"{all_data['_flag_missing_common_name'].sum():,}"
)

print(
    f"Rows with names not in your map: "
    f"{all_data['_flag_name_not_in_name_map'].sum():,}"
)

print(
    f"Possible duplicate rows: "
    f"{len(duplicate_rows):,}"
)

print(
    f"Outside nocturnal window: "
    f"{len(outside_night):,}"
)

print(
    f"Missing sunrise/sunset offsets: "
    f"{len(missing_sun):,}"
)


# ============================================================
# 20. PRINT NEW / DIFFERENT TAXA
# ============================================================

print_section(
    "COMMON NAMES NOT IN YOUR CURRENT name_map"
)

if names_not_in_map.empty:

    print("None.")

else:

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_colwidth",
        None,
        "display.width",
        200,
    ):
        print(
            names_not_in_map
            .to_string(index=False)
        )


print_section(
    "taxonCode VALUES WITH MISSING commonName"
)

if missing_name_taxa.empty:

    print("None.")

else:

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_colwidth",
        None,
        "display.width",
        200,
    ):
        print(
            missing_name_taxa
            .to_string(index=False)
        )


# ============================================================
# 21. OUTPUT FILE LIST
# ============================================================

print_section("OUTPUT FILES")

for path in sorted(
    OUTPUT_DIR.glob("*.csv")
):
    print(path.name)


# ============================================================
# 22. FINAL MESSAGE
# ============================================================

print("\nDONE.")

print(
    "\nIMPORTANT:"
)

print(
    "- Raw files were NOT modified."
)

print(
    "- The master dataset keeps every row."
)

print(
    "- Questionable taxa were flagged, not silently removed."
)

print(
    "- Fall, Spring, and Full-season nocturnal datasets "
    "were exported separately."
)

print(
    "\nNext step:"
)

print(
    "Review taxonomy_review_summary.csv and "
    "names_not_in_name_map.csv, then assign Family1 "
    "and create site_family_calls for each season."
)
