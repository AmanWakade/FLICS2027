import pandas as pd

ROOT_FILE = "KIDPAN_DATA_converted.csv"
MODEL_FILE = "MODEL_DATASET_LEAKAGE_FREE.csv"
OUTPUT_FILE = "MODEL_DATASET_LEAKAGE_FREE_WITH_CENTER.csv"

ID_COL = "TRR_ID_CODE"
CENTER_COL = "CTR_CODE"

print("=" * 70)
print("ADDING TRANSPLANT CENTER CODE")
print("=" * 70)

# ============================================================
# Load final leakage-free model dataset
# ============================================================
model_df = pd.read_csv(
    MODEL_FILE,
    low_memory=False
)

rows_before = len(model_df)

if ID_COL not in model_df.columns:
    raise ValueError(f"{ID_COL} is missing from {MODEL_FILE}")

model_df[ID_COL] = (
    model_df[ID_COL]
    .astype("string")
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
)

wanted_ids = set(
    model_df[ID_COL]
    .dropna()
    .unique()
)

print(f"\nLeakage-free rows : {rows_before:,}")
print(f"Unique patient IDs: {len(wanted_ids):,}")

# ============================================================
# Read only TRR_ID_CODE and CTR_CODE from the large root file
# Use chunks to avoid memory problems
# ============================================================
print("\nReading center information from root KIDPAN file...")

matched_chunks = []

for chunk_number, chunk in enumerate(
    pd.read_csv(
        ROOT_FILE,
        usecols=[ID_COL, CENTER_COL],
        chunksize=100_000,
        low_memory=False,
        on_bad_lines="skip",
    ),
    start=1,
):
    chunk[ID_COL] = (
        chunk[ID_COL]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    chunk = chunk[
        chunk[ID_COL].isin(wanted_ids)
    ]

    if not chunk.empty:
        matched_chunks.append(chunk)

    print(
        f"Chunk {chunk_number}: "
        f"matched rows = {len(chunk):,}"
    )

if not matched_chunks:
    raise ValueError(
        "No matching TRR_ID_CODE values were found."
    )

center_df = pd.concat(
    matched_chunks,
    ignore_index=True
)

print(
    f"\nMatched root rows before deduplication: "
    f"{len(center_df):,}"
)

# ============================================================
# Audit conflicting center codes
# ============================================================
center_df[CENTER_COL] = (
    center_df[CENTER_COL]
    .astype("string")
    .str.strip()
)

center_df[CENTER_COL] = center_df[CENTER_COL].replace(
    {
        "": pd.NA,
        ".": pd.NA,
        "nan": pd.NA,
        "None": pd.NA,
        "<NA>": pd.NA,
    }
)

center_counts = (
    center_df
    .dropna(subset=[CENTER_COL])
    .groupby(ID_COL)[CENTER_COL]
    .nunique()
)

conflicting_ids = center_counts[
    center_counts > 1
]

print(
    "Patients with multiple different CTR_CODE values:",
    len(conflicting_ids),
)

if len(conflicting_ids) > 0:
    conflict_audit = center_df[
        center_df[ID_COL].isin(conflicting_ids.index)
    ].sort_values(ID_COL)

    conflict_audit.to_csv(
        "ctr_code_conflict_audit.csv",
        index=False
    )

    print("Saved: ctr_code_conflict_audit.csv")

# ============================================================
# Keep one center code per transplant
#
# Mode is used when duplicate root rows exist.
# If all values are missing, CTR_CODE remains missing.
# ============================================================
def select_center(series):
    values = series.dropna()

    if values.empty:
        return pd.NA

    return values.mode().iloc[0]


center_unique = (
    center_df
    .groupby(ID_COL, as_index=False)
    .agg({CENTER_COL: select_center})
)

if center_unique[ID_COL].duplicated().any():
    raise ValueError(
        "Duplicate TRR_ID_CODE values remain in center lookup."
    )

print(
    f"Unique center lookup rows: "
    f"{len(center_unique):,}"
)

# ============================================================
# Merge center code into leakage-free model dataset
# ============================================================
final_df = model_df.merge(
    center_unique,
    on=ID_COL,
    how="left",
    validate="one_to_one",
)

rows_after = len(final_df)

if rows_after != rows_before:
    raise ValueError(
        f"Row count changed from {rows_before} to {rows_after}."
    )

# ============================================================
# Validation and summary
# ============================================================
matched = final_df[CENTER_COL].notna().sum()
missing = final_df[CENTER_COL].isna().sum()
unique_centers = final_df[CENTER_COL].nunique(dropna=True)

print("\n" + "=" * 70)
print("MERGE SUMMARY")
print("=" * 70)
print(f"Rows before merge       : {rows_before:,}")
print(f"Rows after merge        : {rows_after:,}")
print(f"Rows with CTR_CODE      : {matched:,}")
print(f"Rows missing CTR_CODE   : {missing:,}")
print(f"Unique transplant centers: {unique_centers:,}")

print("\nLargest centers:")
print(
    final_df[CENTER_COL]
    .value_counts(dropna=False)
    .head(20)
)

# ============================================================
# Save
# ============================================================
final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved: {OUTPUT_FILE}")