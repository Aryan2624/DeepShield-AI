from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DATASET_DIR = Path(
    r"E:\DeepShield-Datasets\phishing_messages"
)

OUTPUT_DIR = (
    RAW_DATASET_DIR
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SOURCE_FILES = [
    "CEAS_08.csv",
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nigerian_Fraud.csv",
    "SpamAssasin.csv",
]


RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================

def normalize_text(series):
    return (
        series
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def load_source(filename):
    path = RAW_DATASET_DIR / filename

    df = pd.read_csv(
        path,
        usecols=[
            "subject",
            "body",
            "label",
        ],
        low_memory=False,
    )

    subject = (
        df["subject"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    body = (
        df["body"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    text = (
        subject
        + " "
        + body
    ).str.strip()

    output = pd.DataFrame(
        {
            "text": text,
            "label": df["label"].astype(int),
            "source": filename.replace(
                ".csv",
                "",
            ),
        }
    )

    return output


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 90)
print("DEEPSHIELD AI - PHISHING DATASET BUILDER V1")
print("=" * 90)


frames = []


for filename in SOURCE_FILES:
    df = load_source(filename)

    frames.append(df)

    print(
        f"{filename:<25}"
        f"{len(df):>8,} rows"
    )


combined = pd.concat(
    frames,
    ignore_index=True,
)


print("\n" + "=" * 90)
print("RAW COMBINED DATASET")
print("=" * 90)

print(
    f"Total rows: "
    f"{len(combined):,}"
)


# ============================================================
# NORMALIZE FOR DEDUPLICATION
# ============================================================

combined["normalized_text"] = normalize_text(
    combined["text"]
)


# Remove empty text just in case
empty_mask = (
    combined["normalized_text"]
    .eq("")
)

empty_count = empty_mask.sum()

combined = combined[
    ~empty_mask
].copy()


print(
    f"Empty messages removed: "
    f"{empty_count:,}"
)


# ============================================================
# CHECK LABEL CONFLICTS
# ============================================================

label_counts = (
    combined
    .groupby("normalized_text")["label"]
    .nunique()
)

conflicting_texts = (
    label_counts[
        label_counts > 1
    ]
    .index
)


print(
    f"Conflicting messages: "
    f"{len(conflicting_texts):,}"
)


if len(conflicting_texts) > 0:
    print(
        "Removing messages with conflicting labels..."
    )

    combined = combined[
        ~combined["normalized_text"]
        .isin(conflicting_texts)
    ].copy()


# ============================================================
# REMOVE DUPLICATES
# ============================================================

rows_before_dedup = len(combined)


combined = (
    combined
    .drop_duplicates(
        subset="normalized_text",
        keep="first",
    )
    .reset_index(drop=True)
)


duplicates_removed = (
    rows_before_dedup
    - len(combined)
)


print(
    f"Duplicate rows removed: "
    f"{duplicates_removed:,}"
)


print(
    f"Final unique messages: "
    f"{len(combined):,}"
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\nFinal label distribution:")

print(
    combined["label"]
    .value_counts()
    .sort_index()
)


print("\nFinal label percentages:")

print(
    combined["label"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


print("\nFinal source distribution:")

print(
    combined["source"]
    .value_counts()
)


# ============================================================
# STRATIFICATION KEY
# ============================================================

# Preserves both source distribution and class distribution
combined["stratify_key"] = (
    combined["source"].astype(str)
    + "__"
    + combined["label"].astype(str)
)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_df, temp_df = train_test_split(
    combined,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=combined["stratify_key"],
)


validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["stratify_key"],
)


train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# LEAKAGE CHECK
# ============================================================

train_texts = set(
    train_df["normalized_text"]
)

validation_texts = set(
    validation_df["normalized_text"]
)

test_texts = set(
    test_df["normalized_text"]
)


train_validation_overlap = len(
    train_texts
    & validation_texts
)

train_test_overlap = len(
    train_texts
    & test_texts
)

validation_test_overlap = len(
    validation_texts
    & test_texts
)


print("\n" + "=" * 90)
print("SPLIT SUMMARY")
print("=" * 90)


print(
    f"Train rows:      "
    f"{len(train_df):,}"
)

print(
    f"Validation rows: "
    f"{len(validation_df):,}"
)

print(
    f"Test rows:       "
    f"{len(test_df):,}"
)


print("\nLeakage check:")

print(
    f"Train ↔ Validation overlap: "
    f"{train_validation_overlap}"
)

print(
    f"Train ↔ Test overlap: "
    f"{train_test_overlap}"
)

print(
    f"Validation ↔ Test overlap: "
    f"{validation_test_overlap}"
)


# ============================================================
# DISPLAY CLASS DISTRIBUTION PER SPLIT
# ============================================================

for name, df in [
    ("TRAIN", train_df),
    ("VALIDATION", validation_df),
    ("TEST", test_df),
]:
    print(
        "\n"
        + "-" * 90
    )

    print(name)

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    print(
        df["label"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )


# ============================================================
# REMOVE INTERNAL COLUMNS BEFORE SAVING
# ============================================================

save_columns = [
    "text",
    "label",
    "source",
]


master_output = combined[
    save_columns
].copy()

train_output = train_df[
    save_columns
].copy()

validation_output = validation_df[
    save_columns
].copy()

test_output = test_df[
    save_columns
].copy()


# ============================================================
# SAVE DATASETS
# ============================================================

master_path = (
    OUTPUT_DIR
    / "phishing_master_v1.csv"
)

train_path = (
    OUTPUT_DIR
    / "phishing_train_v1.csv"
)

validation_path = (
    OUTPUT_DIR
    / "phishing_validation_v1.csv"
)

test_path = (
    OUTPUT_DIR
    / "phishing_test_v1.csv"
)


master_output.to_csv(
    master_path,
    index=False,
)

train_output.to_csv(
    train_path,
    index=False,
)

validation_output.to_csv(
    validation_path,
    index=False,
)

test_output.to_csv(
    test_path,
    index=False,
)


print("\n" + "=" * 90)
print("FILES SAVED")
print("=" * 90)

print(master_path)
print(train_path)
print(validation_path)
print(test_path)


print("\n" + "=" * 90)
print("DATASET BUILD COMPLETE")
print("=" * 90)