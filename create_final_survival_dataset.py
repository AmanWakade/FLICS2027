import pandas as pd
import numpy as np

# =====================================================
# Input files
# =====================================================
kidpan_file = "KIDPAN_DATA_converted.csv"
malig_file = "kidpan_malig_followup_converted_full.csv"
output_file = "FINAL_SURVIVAL_DATASET.csv"

# =====================================================
# Clinical features from KIDPAN
# =====================================================
kidpan_cols = [
    "TRR_ID_CODE",

    # Recipient
    "AGE", "GENDER", "ABO", "ETHNICITY", "BMI_CALC",
    "DIAB", "DGN_TCR", "ON_DIALYSIS", "PREV_TX",
    "PREV_KI_TX", "FUNC_STAT_TRR",

    # Immunological
    "CURRENT_PRA", "PEAK_PRA", "INIT_CPRA", "END_CPRA",
    "AMIS", "BMIS", "DRMIS", "HLAMIS", "ABO_MAT",

    # Donor
    "AGE_DON", "GENDER_DON", "ABO_DON", "DON_TY",
    "KDPI", "KDRI_MED", "BMI_DON_CALC", "CREAT_DON",
    "BUN_DON", "DIABETES_DON", "HIST_DIABETES_DON",
    "HIST_HYPERTENS_DON", "CMV_DON", "EBV_IGG_DON",
    "HBV_NAT_DON", "HCV_NAT_DON",

    # Transplant
    "TX_DATE", "COLD_ISCH_KI", "WARM_ISCH_TM_DON",
    "DISTANCE", "PUMP_KI", "MULTIORG", "TX_TYPE",
    "SHARE_TY",

    # Optional early post-transplant
    "ACUTE_REJ_EPI_KI", "CREAT_TRR", "CREAT6M",
    "CREAT1Y", "FIRST_WK_DIAL", "GRF_STAT_KI", "LOS"
]

# =====================================================
# Malignancy columns
# =====================================================
dx_cols = [f"DX_DATE{i}" for i in range(1, 30)]

malig_cols = [
    "TRR_ID_CODE",
    "MALIG",
    "PX_STAT_DATE"
] + dx_cols + ["DX_DATE_LYMPH"]

# =====================================================
# Load malignancy file first
# =====================================================
print("Loading malignancy file...")
malig = pd.read_csv(
    malig_file,
    usecols=lambda c: c in malig_cols,
    low_memory=False
)

malig["TRR_ID_CODE"] = malig["TRR_ID_CODE"].astype(str)
wanted_ids = set(malig["TRR_ID_CODE"].dropna().unique())

print("Malignancy rows:", malig.shape)
print("Unique TRR_ID_CODE in malignancy:", len(wanted_ids))

# =====================================================
# Load KIDPAN in chunks and keep only matching TRR_ID_CODE
# =====================================================
print("\nLoading KIDPAN selected features in chunks...")

kidpan_chunks = []

for i, chunk in enumerate(pd.read_csv(
    kidpan_file,
    usecols=lambda c: c in kidpan_cols,
    chunksize=50000,
    engine="python",
    on_bad_lines="skip"
)):
    chunk["TRR_ID_CODE"] = chunk["TRR_ID_CODE"].astype(str)
    chunk = chunk[chunk["TRR_ID_CODE"].isin(wanted_ids)]
    kidpan_chunks.append(chunk)
    print(f"Chunk {i+1} processed, kept rows: {len(chunk)}")

kidpan = pd.concat(kidpan_chunks, ignore_index=True)

# One clinical row per TRR_ID_CODE
kidpan = kidpan.drop_duplicates(subset="TRR_ID_CODE", keep="first")

print("KIDPAN selected shape:", kidpan.shape)

# =====================================================
# Merge malignancy + KIDPAN
# Keep only rows from malignancy file
# =====================================================
print("\nMerging files...")

df = malig.merge(
    kidpan,
    on="TRR_ID_CODE",
    how="left"
)

print("Merged shape:", df.shape)

# =====================================================
# Convert dates
# =====================================================
date_cols = ["TX_DATE", "PX_STAT_DATE", "DX_DATE_LYMPH"] + dx_cols

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# =====================================================
# Create EVENT
# EVENT = 1 if malignancy occurred
# EVENT = 0 if censored/no malignancy during follow-up
# =====================================================
df["EVENT"] = (
    df["MALIG"]
    .astype(str)
    .str.upper()
    .str.strip()
    .eq("Y")
    .astype(int)
)

# =====================================================
# Earliest malignancy date
# =====================================================
available_dx_cols = [c for c in dx_cols + ["DX_DATE_LYMPH"] if c in df.columns]

df["FIRST_EVENT_DATE"] = df[available_dx_cols].min(axis=1)

# =====================================================
# Create TIME
# If EVENT = 1: FIRST_EVENT_DATE - TX_DATE
# If EVENT = 0: PX_STAT_DATE - TX_DATE
# =====================================================
df["TIME_DAYS"] = np.where(
    df["EVENT"] == 1,
    (df["FIRST_EVENT_DATE"] - df["TX_DATE"]).dt.days,
    (df["PX_STAT_DATE"] - df["TX_DATE"]).dt.days
)

df["TIME_YEARS"] = df["TIME_DAYS"] / 365.25

# =====================================================
# Cleaning invalid survival rows
# =====================================================
initial_rows = len(df)

df = df[df["TX_DATE"].notna()]
df = df[df["TIME_DAYS"].notna()]
df = df[df["TIME_DAYS"] > 0]

final_rows = len(df)

# =====================================================
# Remove raw diagnosis/date outcome columns from final predictors
# Keep only EVENT and TIME_YEARS as labels
# =====================================================
drop_cols = dx_cols + [
    "DX_DATE_LYMPH",
    "PX_STAT_DATE",
    "MALIG",
    "FIRST_EVENT_DATE",
    "TIME_DAYS"
]

df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Put labels at end
label_cols = ["EVENT", "TIME_YEARS"]
feature_cols = [c for c in df.columns if c not in label_cols]
df = df[feature_cols + label_cols]

# =====================================================
# Save final dataset
# =====================================================
df.to_csv(output_file, index=False)

# =====================================================
# Summary
# =====================================================
print("\n" + "=" * 60)
print("FINAL DATASET SUMMARY")
print("=" * 60)
print("Initial merged rows :", initial_rows)
print("Final rows          :", final_rows)
print("Removed rows        :", initial_rows - final_rows)
print("Final columns       :", df.shape[1])
print("Saved file          :", output_file)

print("\nEVENT counts:")
print(df["EVENT"].value_counts())

print("\nMedian TIME_YEARS:", round(df["TIME_YEARS"].median(), 2))
print("Max TIME_YEARS   :", round(df["TIME_YEARS"].max(), 2))