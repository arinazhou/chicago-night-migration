"""
build_family_call_counts.py

Create model-ready family call-count datasets for:
    1. Fall 2025
    2. Spring 2026
    3. Full season = Fall 2025 + Spring 2026

IMPORTANT SAFETY CHOICES
------------------------
- Uses ONLY detections whose commonName is explicitly listed in FAMILY_MAP.
- Does NOT count family-level detector outputs such as "Parulidae",
  "Turdidae", etc., because those could double-count events already
  represented by species-level detections.
- Does NOT force ambiguous codes such as gcbi, zeep, dbup, thsh, etc.
  into a family.
- Saves excluded/unmapped rows for manual review.
- Preserves the original site name in clean detection exports.
- Creates a separate `site_model` column for analysis-site renaming.
- Produces BOTH:
      observed-only family counts
      zero-filled site x family counts
  The zero-filled version is usually the one to use for count models,
  because a family that was not detected at a sampled site should have
  n_calls = 0 rather than disappearing from the dataset.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

# Script location:
# bird/data/family_calls_data/build_family_call_counts.py

FAMILY_DIR = Path(__file__).resolve().parent

INPUT_DIR = (
    FAMILY_DIR
    / "family_call_preprocessing_output"
)

OUTPUT_DIR = (
    FAMILY_DIR
    / "family_call_model_data"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FALL_FILE = (
    INPUT_DIR
    / "fall2025_nocturnal_detections.csv"
)

SPRING_FILE = (
    INPUT_DIR
    / "spring2026_nocturnal_detections.csv"
)


# ============================================================
# 2. OPTIONAL SITE RENAMING FOR ANALYSIS
# ============================================================

# Do NOT overwrite the raw/original `site` column.
# Instead, create `site_model` for merging with your newer
# environmental-predictor tables.

SITE_RENAME = {
    "ALTGELD-MARINA": "DOLTON-PIER11",
}


# ============================================================
# 3. APPROVED SPECIES -> FAMILY MAP
# ============================================================

# Only species explicitly listed here will contribute to family counts.
# If a new species appears later, it will be exported for review instead
# of being silently assigned or deleted.

FAMILY_MAP = {

    # --------------------------------------------------------
    # Caprimulgidae
    # --------------------------------------------------------
    "Common Nighthawk": "Caprimulgidae",

    # --------------------------------------------------------
    # Turdidae
    # --------------------------------------------------------
    "Swainson's Thrush": "Turdidae",
    "American Robin": "Turdidae",
    "Wood Thrush": "Turdidae",
    "Gray-cheeked Thrush": "Turdidae",
    "Veery": "Turdidae",
    "Hermit Thrush": "Turdidae",
    "Eastern Bluebird": "Turdidae",

    # --------------------------------------------------------
    # Cardinalidae
    # --------------------------------------------------------
    "Rose-breasted Grosbeak": "Cardinalidae",
    "Indigo Bunting": "Cardinalidae",
    "Dickcissel": "Cardinalidae",
    "Scarlet Tanager": "Cardinalidae",
    "Blue Grosbeak": "Cardinalidae",

    # --------------------------------------------------------
    # Passerellidae
    # --------------------------------------------------------
    "Savannah Sparrow": "Passerellidae",
    "White-crowned Sparrow": "Passerellidae",
    "White-throated Sparrow": "Passerellidae",
    "Chipping Sparrow": "Passerellidae",
    "Lincoln's Sparrow": "Passerellidae",
    "Grasshopper Sparrow": "Passerellidae",
    "Dark-eyed Junco": "Passerellidae",
    "American Tree Sparrow": "Passerellidae",
    "LeConte's Sparrow": "Passerellidae",
    "Vesper Sparrow": "Passerellidae",
    "Brewer's Sparrow": "Passerellidae",
    "Song Sparrow": "Passerellidae",
    "Lark Sparrow": "Passerellidae",
    "Field Sparrow": "Passerellidae",

    # --------------------------------------------------------
    # Parulidae
    # --------------------------------------------------------
    "Ovenbird": "Parulidae",
    "Palm Warbler": "Parulidae",
    "American Redstart": "Parulidae",
    "Cape May Warbler": "Parulidae",
    "Common Yellowthroat": "Parulidae",
    "Black-and-white Warbler": "Parulidae",
    "Magnolia Warbler": "Parulidae",
    "Northern Waterthrush": "Parulidae",
    "Northern Parula": "Parulidae",
    "Chestnut-sided Warbler": "Parulidae",
    "Yellow-rumped Warbler": "Parulidae",
    "Mourning Warbler": "Parulidae",
    "Hooded Warbler": "Parulidae",
    "Canada Warbler": "Parulidae",
    "Wilson's Warbler": "Parulidae",
    "Nashville Warbler": "Parulidae",

    # --------------------------------------------------------
    # Motacillidae
    # --------------------------------------------------------
    "American Pipit": "Motacillidae",
    "Sprague's Pipit": "Motacillidae",

    # --------------------------------------------------------
    # Sittidae
    # --------------------------------------------------------
    "Red-breasted Nuthatch": "Sittidae",

    # --------------------------------------------------------
    # Ardeidae
    # --------------------------------------------------------
    "Green Heron": "Ardeidae",
    "Black-crowned Night Heron": "Ardeidae",
    "Great Blue Heron": "Ardeidae",
    "American Bittern": "Ardeidae",
    "Least Bittern": "Ardeidae",
    "Yellow-crowned Night Heron": "Ardeidae",
    "Western Cattle Egret": "Ardeidae",

    # --------------------------------------------------------
    # Charadriidae
    # --------------------------------------------------------
    "Killdeer": "Charadriidae",
    "American Golden-Plover": "Charadriidae",
    "Semipalmated Plover": "Charadriidae",
    "Black-bellied Plover": "Charadriidae",
    "Piping Plover": "Charadriidae",

    # --------------------------------------------------------
    # Icteridae
    # --------------------------------------------------------
    "Bobolink": "Icteridae",
    "Eastern Meadowlark": "Icteridae",

    # --------------------------------------------------------
    # Cuculidae
    # --------------------------------------------------------
    "Black-billed Cuckoo": "Cuculidae",
    "Yellow-billed Cuckoo": "Cuculidae",

    # --------------------------------------------------------
    # Scolopacidae
    # --------------------------------------------------------
    "Wilson's Snipe": "Scolopacidae",
    "Least Sandpiper": "Scolopacidae",
    "Solitary Sandpiper": "Scolopacidae",
    "Spotted Sandpiper": "Scolopacidae",
    "Baird's Sandpiper": "Scolopacidae",
    "Upland Sandpiper": "Scolopacidae",
    "Red Knot": "Scolopacidae",
    "Greater Yellowlegs": "Scolopacidae",
    "Dunlin": "Scolopacidae",
    "Ruddy Turnstone": "Scolopacidae",
    "Lesser Yellowlegs": "Scolopacidae",
    "Long-billed Curlew": "Scolopacidae",
    "Wilson's Phalarope": "Scolopacidae",
    "Short-billed Dowitcher": "Scolopacidae",
    "Sanderling": "Scolopacidae",
    "Whimbrel": "Scolopacidae",
    "Willet": "Scolopacidae",

    # --------------------------------------------------------
    # Regulidae
    # --------------------------------------------------------
    "Golden-crowned Kinglet": "Regulidae",

    # --------------------------------------------------------
    # Laridae
    # --------------------------------------------------------
    "Caspian Tern": "Laridae",

    # --------------------------------------------------------
    # Bombycillidae
    # --------------------------------------------------------
    "Cedar Waxwing": "Bombycillidae",

    # --------------------------------------------------------
    # Anatidae
    # --------------------------------------------------------
    "Canada Goose": "Anatidae",
    "Black-bellied Whistling-Duck": "Anatidae",
    "Brant": "Anatidae",
    "Tundra Swan": "Anatidae",
    "Green-winged Teal": "Anatidae",
    "Northern Pintail": "Anatidae",

    # --------------------------------------------------------
    # Fringillidae
    # --------------------------------------------------------
    "Pine Siskin": "Fringillidae",
    "Purple Finch": "Fringillidae",

    # --------------------------------------------------------
    # Gruidae
    # --------------------------------------------------------
    "Sandhill Crane": "Gruidae",

    # --------------------------------------------------------
    # Calcariidae
    # --------------------------------------------------------
    "Lapland Longspur": "Calcariidae",
    "Snow Bunting": "Calcariidae",

    # --------------------------------------------------------
    # Rallidae
    # --------------------------------------------------------
    "Virginia Rail": "Rallidae",
    "Sora": "Rallidae",

    # --------------------------------------------------------
    # Alaudidae
    # --------------------------------------------------------
    "Horned Lark": "Alaudidae",
}


# ============================================================
# 4. HELPERS
# ============================================================

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_season(path, season_label):
    """
    Load one season's nocturnal detections and add explicit season label.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"\nCould not find:\n{path}\n\n"
            "Run prepare_family_call_data_safe.py first."
        )

    df = pd.read_csv(
        path,
        low_memory=False
    )

    required = [
        "site",
        "commonName",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {missing}"
        )

    df = df.copy()

    df["season"] = season_label

    # Preserve original site.
    df["site_model"] = (
        df["site"]
        .replace(SITE_RENAME)
    )

    return df


def split_approved_and_review(df):
    """
    Use only explicitly approved species names for the count dataset.

    Everything else remains available in a review table.
    """

    df = df.copy()

    df["Family1"] = (
        df["commonName"]
        .map(FAMILY_MAP)
    )

    approved = (
        df.loc[
            df["Family1"].notna()
        ]
        .copy()
    )

    review = (
        df.loc[
            df["Family1"].isna()
        ]
        .copy()
    )

    return approved, review


def make_observed_counts(clean_df):
    """
    Count actual detection rows by site and family.

    Only combinations with >= 1 detection appear.
    """

    counts = (
        clean_df
        .groupby(
            ["site_model", "Family1"],
            as_index=False
        )
        .size()
        .rename(
            columns={
                "size": "n_calls"
            }
        )
    )

    return counts


def make_zero_filled_counts(clean_df, all_sampled_sites):
    """
    Build complete site x family grid and fill non-detections with 0.

    This is generally the safer format for family-specific count models,
    because absence at a sampled site should be represented as n_calls = 0.
    """

    observed = make_observed_counts(
        clean_df
    )

    families = sorted(
        clean_df["Family1"]
        .dropna()
        .unique()
    )

    sites = sorted(
        pd.Series(all_sampled_sites)
        .dropna()
        .unique()
    )

    grid = pd.MultiIndex.from_product(
        [
            sites,
            families
        ],
        names=[
            "site_model",
            "Family1"
        ]
    ).to_frame(
        index=False
    )

    complete = (
        grid
        .merge(
            observed,
            on=[
                "site_model",
                "Family1"
            ],
            how="left"
        )
    )

    complete["n_calls"] = (
        complete["n_calls"]
        .fillna(0)
        .astype(int)
    )

    return complete


def save_season_outputs(
    raw_df,
    clean_df,
    review_df,
    season_name
):
    """
    Save detection-level and count-level outputs for one season.
    """

    # --------------------------------------------------------
    # Detection-level clean rows
    # --------------------------------------------------------

    clean_path = (
        OUTPUT_DIR
        / f"{season_name}_clean_species_detections.csv"
    )

    clean_df.to_csv(
        clean_path,
        index=False
    )


    # --------------------------------------------------------
    # Unmapped/review rows
    # --------------------------------------------------------

    review_path = (
        OUTPUT_DIR
        / f"{season_name}_excluded_for_manual_review.csv"
    )

    review_df.to_csv(
        review_path,
        index=False
    )


    # --------------------------------------------------------
    # Observed-only counts
    # --------------------------------------------------------

    observed_counts = make_observed_counts(
        clean_df
    )

    observed_path = (
        OUTPUT_DIR
        / f"site_family_calls_{season_name}_observed_only.csv"
    )

    observed_counts.to_csv(
        observed_path,
        index=False
    )


    # --------------------------------------------------------
    # Zero-filled counts
    # --------------------------------------------------------

    sampled_sites = (
        raw_df["site_model"]
        .dropna()
        .unique()
    )

    complete_counts = make_zero_filled_counts(
        clean_df,
        sampled_sites
    )

    complete_path = (
        OUTPUT_DIR
        / f"site_family_calls_{season_name}.csv"
    )

    complete_counts.to_csv(
        complete_path,
        index=False
    )


    return {
        "observed": observed_counts,
        "complete": complete_counts,
        "clean": clean_df,
        "review": review_df,
    }


# ============================================================
# 5. LOAD FALL AND SPRING
# ============================================================

print_section("LOAD SEASON DATA")

fall = load_season(
    FALL_FILE,
    "fall2025"
)

spring = load_season(
    SPRING_FILE,
    "spring2026"
)

print(
    f"Fall detections loaded:   {len(fall):,}"
)

print(
    f"Spring detections loaded: {len(spring):,}"
)


# ============================================================
# 6. BUILD FULL-SEASON DETECTION DATA
# ============================================================

print_section("BUILD FULL SEASON")

fullseason = pd.concat(
    [
        fall,
        spring
    ],
    ignore_index=True
)

assert (
    len(fullseason)
    == len(fall) + len(spring)
)

print(
    f"Full-season detections:   {len(fullseason):,}"
)


# ============================================================
# 7. SPLIT APPROVED SPECIES / REVIEW ROWS
# ============================================================

print_section("ASSIGN FAMILY1")

fall_clean, fall_review = (
    split_approved_and_review(fall)
)

spring_clean, spring_review = (
    split_approved_and_review(spring)
)

full_clean, full_review = (
    split_approved_and_review(fullseason)
)


print(
    f"Fall approved species rows:   {len(fall_clean):,}"
)

print(
    f"Fall review/excluded rows:    {len(fall_review):,}"
)

print(
    f"Spring approved species rows: {len(spring_clean):,}"
)

print(
    f"Spring review/excluded rows:  {len(spring_review):,}"
)

print(
    f"Full approved species rows:   {len(full_clean):,}"
)

print(
    f"Full review/excluded rows:    {len(full_review):,}"
)


# ============================================================
# 8. SAVE SEASON OUTPUTS
# ============================================================

print_section("CREATE FAMILY COUNT DATASETS")

fall_out = save_season_outputs(
    raw_df=fall,
    clean_df=fall_clean,
    review_df=fall_review,
    season_name="fall2025"
)

spring_out = save_season_outputs(
    raw_df=spring,
    clean_df=spring_clean,
    review_df=spring_review,
    season_name="spring2026"
)

full_out = save_season_outputs(
    raw_df=fullseason,
    clean_df=full_clean,
    review_df=full_review,
    season_name="fullseason"
)


# ============================================================
# 9. FAMILY SUMMARY
# ============================================================

print_section("FAMILY SUMMARY")

summary_rows = []

for season_name, result in [
    ("fall2025", fall_out),
    ("spring2026", spring_out),
    ("fullseason", full_out),
]:

    clean = result["clean"]

    family_summary = (
        clean
        .groupby("Family1")
        .agg(
            n_detections=(
                "Family1",
                "size"
            ),
            n_sites=(
                "site_model",
                "nunique"
            ),
            n_species=(
                "commonName",
                "nunique"
            )
        )
        .reset_index()
    )

    family_summary.insert(
        0,
        "season",
        season_name
    )

    summary_rows.append(
        family_summary
    )


family_summary_all = pd.concat(
    summary_rows,
    ignore_index=True
)

family_summary_all.to_csv(
    OUTPUT_DIR
    / "family_summary_by_season.csv",
    index=False
)


# ============================================================
# 10. REVIEW SUMMARY FOR EXCLUDED ROWS
# ============================================================

print_section("EXCLUDED TAXA SUMMARY")

review_summary = (
    full_review
    .groupby(
        [
            "commonName",
            "taxonCode"
        ],
        dropna=False
    )
    .size()
    .reset_index(
        name="n_rows"
    )
    .sort_values(
        "n_rows",
        ascending=False
    )
)

review_summary.to_csv(
    OUTPUT_DIR
    / "excluded_taxa_summary_fullseason.csv",
    index=False
)

with pd.option_context(
    "display.max_rows",
    100,
    "display.width",
    180,
    "display.max_colwidth",
    80,
):
    print(
        review_summary
        .head(100)
        .to_string(index=False)
    )


# ============================================================
# 11. SAFETY / CONSISTENCY CHECKS
# ============================================================

print_section("SAFETY CHECKS")

# Approved + review must recover every row.
assert (
    len(fall_clean)
    + len(fall_review)
    == len(fall)
)

assert (
    len(spring_clean)
    + len(spring_review)
    == len(spring)
)

assert (
    len(full_clean)
    + len(full_review)
    == len(fullseason)
)

# Full clean data should equal Fall clean + Spring clean.
assert (
    len(full_clean)
    == len(fall_clean)
    + len(spring_clean)
)

# Full call total should equal number of approved detection rows.
assert (
    full_out["observed"]["n_calls"].sum()
    == len(full_clean)
)

print(
    "All row-preservation checks passed."
)

print(
    "Full-season family counts equal "
    "the number of approved species detections."
)


# ============================================================
# 12. PRINT MODEL-READY FILES
# ============================================================

print_section("MODEL-READY FILES")

print(
    "\nUse these for your family count models:"
)

print(
    OUTPUT_DIR
    / "site_family_calls_fall2025.csv"
)

print(
    OUTPUT_DIR
    / "site_family_calls_spring2026.csv"
)

print(
    OUTPUT_DIR
    / "site_family_calls_fullseason.csv"
)

print(
    "\nThese are ZERO-FILLED site x family datasets."
)

print(
    "Example:"
)

print(
    full_out["complete"]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print_section("DONE")

print(
    "Nothing was silently discarded."
)

print(
    "Rows that were not explicitly approved in FAMILY_MAP "
    "were saved to manual-review CSVs."
)

print(
    "The three site_family_calls_*.csv files are the "
    "model-ready family count tables."
)
