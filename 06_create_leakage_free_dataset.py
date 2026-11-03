from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path("MODEL_DATASET.csv")
OUTPUT_FILE = Path("MODEL_DATASET_LEAKAGE_FREE.csv")
AUDIT_FILE = Path("duplicate_transplant_audit.csv")

ID_COL = "TRR_ID_CODE"
EVENT_COL = "EVENT"
TIME_COL = "TIME_YEARS"


# ============================================================
# Load dataset
# ============================================================

print("=" * 70)
print("CREATING LEAKAGE-FREE MODEL DATASET")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"File not found: {INPUT_FILE.resolve()}"
    )

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("\nOriginal shape:", df.shape)

required_columns = {
    ID_COL,
    EVENT_COL,
    TIME_COL,
}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"Required columns missing: {sorted(missing_columns)}"
    )


# ============================================================
# Clean transplant ID
# ============================================================

df[ID_COL] = (
    df[ID_COL]
    .astype("string")
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
)

df[ID_COL] = df[ID_COL].replace(
    {
        "": pd.NA,
        ".": pd.NA,
        "nan": pd.NA,
        "None": pd.NA,
        "<NA>": pd.NA,
    }
)


# ============================================================
# Clean survival outcomes
# ============================================================

df[EVENT_COL] = pd.to_numeric(
    df[EVENT_COL],
    errors="coerce",
)

df[TIME_COL] = pd.to_numeric(
    df[TIME_COL],
    errors="coerce",
)

valid_mask = (
    df[ID_COL].notna()
    & df[EVENT_COL].isin([0, 1])
    & df[TIME_COL].notna()
    & np.isfinite(df[TIME_COL])
    & (df[TIME_COL] > 0)
)

invalid_rows = int((~valid_mask).sum())

df = df.loc[valid_mask].copy()

df[EVENT_COL] = df[EVENT_COL].astype(np.uint8)
df[TIME_COL] = df[TIME_COL].astype(np.float32)

print("Invalid rows removed:", invalid_rows)
print("Valid shape:", df.shape)

print("\nRow-level event distribution:")
print(df[EVENT_COL].value_counts().sort_index())

print(
    f"Row-level event rate: "
    f"{df[EVENT_COL].mean() * 100:.2f}%"
)


# ============================================================
# Check duplicate transplant IDs
# ============================================================

id_counts = df[ID_COL].value_counts()

duplicate_counts = id_counts[id_counts > 1]

print("\nTransplant ID analysis")
print("----------------------")
print("Unique transplant IDs:", df[ID_COL].nunique())
print("IDs with repeated rows:", len(duplicate_counts))
print(
    "Rows belonging to repeated IDs:",
    int(duplicate_counts.sum()),
)
print(
    "Maximum rows for one transplant:",
    int(id_counts.max()),
)


# ============================================================
# Create audit information
# ============================================================

audit = (
    df.groupby(ID_COL, sort=False)
    .agg(
        NUMBER_OF_ROWS=(ID_COL, "size"),
        NUMBER_OF_EVENT_ROWS=(EVENT_COL, "sum"),
        MIN_TIME_YEARS=(TIME_COL, "min"),
        MAX_TIME_YEARS=(TIME_COL, "max"),
        UNIQUE_EVENT_VALUES=(EVENT_COL, "nunique"),
        UNIQUE_TIME_VALUES=(TIME_COL, "nunique"),
    )
    .reset_index()
)

audit = audit[
    audit["NUMBER_OF_ROWS"] > 1
].copy()

audit.to_csv(AUDIT_FILE, index=False)


# ============================================================
# Select one definitive row per transplant
# ============================================================

def choose_final_record(
    group: pd.DataFrame,
) -> pd.Series:
    """
    For EVENT=1:
        retain the earliest malignancy event record.

    For EVENT=0:
        retain the latest censored follow-up record.
    """

    event_rows = group[group[EVENT_COL] == 1]

    if not event_rows.empty:
        selected_index = event_rows[TIME_COL].idxmin()
        selected_row = group.loc[selected_index].copy()

        selected_row[EVENT_COL] = 1
        selected_row[TIME_COL] = event_rows[TIME_COL].min()

    else:
        selected_index = group[TIME_COL].idxmax()
        selected_row = group.loc[selected_index].copy()

        selected_row[EVENT_COL] = 0
        selected_row[TIME_COL] = group[TIME_COL].max()

    return selected_row


print("\nSelecting one record per transplant...")

unique_df = (
    df.groupby(
        ID_COL,
        sort=False,
        group_keys=False,
    )
    .apply(
        choose_final_record,
        include_groups=False,
    )
    .reset_index()
)


# ============================================================
# Restore data types
# ============================================================

unique_df[ID_COL] = unique_df[ID_COL].astype("string")
unique_df[EVENT_COL] = unique_df[EVENT_COL].astype(np.uint8)
unique_df[TIME_COL] = unique_df[TIME_COL].astype(np.float32)


# ============================================================
# Final validation
# ============================================================

print("\nValidating leakage-free dataset...")

if unique_df[ID_COL].duplicated().any():
    raise ValueError(
        "Duplicate TRR_ID_CODE values remain."
    )

if not unique_df[EVENT_COL].isin([0, 1]).all():
    raise ValueError(
        "EVENT contains invalid values."
    )

if unique_df[TIME_COL].isna().any():
    raise ValueError(
        "TIME_YEARS contains missing values."
    )

if (unique_df[TIME_COL] <= 0).any():
    raise ValueError(
        "TIME_YEARS contains non-positive values."
    )

if not np.isfinite(unique_df[TIME_COL]).all():
    raise ValueError(
        "TIME_YEARS contains infinite values."
    )

if len(unique_df) != unique_df[ID_COL].nunique():
    raise ValueError(
        "Final row count does not match unique ID count."
    )

print("Validation passed.")


# ============================================================
# Display final statistics
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE-FREE DATASET SUMMARY")
print("=" * 70)

print("Original valid rows :", f"{len(df):,}")
print("Final patient rows  :", f"{len(unique_df):,}")
print(
    "Unique transplant IDs:",
    f"{unique_df[ID_COL].nunique():,}",
)

print("\nFinal event distribution:")
print(unique_df[EVENT_COL].value_counts().sort_index())

print(
    f"Final event rate: "
    f"{unique_df[EVENT_COL].mean() * 100:.2f}%"
)

print("\nTIME_YEARS summary:")
print(unique_df[TIME_COL].describe())


# ============================================================
# Save final dataset
# ============================================================

unique_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\nSaved:")
print(f"- {OUTPUT_FILE}")
print(f"- {AUDIT_FILE}")