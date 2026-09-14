from pathlib import Path

import pandas as pd


DATASET_DIR = Path(r"E:\DeepShield-Datasets\phishing_messages")

SOURCE_FILES = [
    "CEAS_08.csv",
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nigerian_Fraud.csv",
    "SpamAssasin.csv",
]


def normalize_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


frames = []

print("=" * 90)
print("DEEPSHIELD AI - GLOBAL PHISHING DATASET AUDIT")
print("=" * 90)


for filename in SOURCE_FILES:
    path = DATASET_DIR / filename

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    subject = df["subject"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)

    combined_text = (
        subject.str.strip()
        + " "
        + body.str.strip()
    ).str.strip()

    temp = pd.DataFrame(
        {
            "text": combined_text,
            "label": df["label"],
            "source": filename.replace(".csv", ""),
        }
    )

    frames.append(temp)

    print(
        f"{filename:<25} "
        f"Rows: {len(temp):>7,}"
    )


combined = pd.concat(
    frames,
    ignore_index=True,
)

combined["normalized_text"] = normalize_text(
    combined["text"]
)


print("\n" + "=" * 90)
print("COMBINED DATASET")
print("=" * 90)

print(
    f"Total rows: "
    f"{len(combined):,}"
)

print(
    f"Unique normalized messages: "
    f"{combined['normalized_text'].nunique():,}"
)

empty_count = (
    combined["normalized_text"]
    .eq("")
    .sum()
)

print(
    f"Empty messages: "
    f"{empty_count:,}"
)


duplicate_rows = (
    combined["normalized_text"]
    .duplicated()
    .sum()
)

print(
    f"Duplicate rows after first occurrence: "
    f"{duplicate_rows:,}"
)


duplicate_groups = (
    combined[
        combined["normalized_text"]
        .duplicated(keep=False)
    ]
    .groupby("normalized_text")
    .size()
)

print(
    f"Duplicate message groups: "
    f"{len(duplicate_groups):,}"
)


label_counts_per_message = (
    combined
    .groupby("normalized_text")["label"]
    .nunique()
)

conflicting_messages = (
    label_counts_per_message[
        label_counts_per_message > 1
    ]
)

print(
    f"Messages with conflicting labels: "
    f"{len(conflicting_messages):,}"
)


print("\nLabel distribution before cleaning:")

print(
    combined["label"]
    .value_counts()
    .sort_index()
)


print("\nLabel percentages:")

print(
    combined["label"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


print("\nSource distribution:")

print(
    combined["source"]
    .value_counts()
)


if len(conflicting_messages) > 0:
    print("\n" + "=" * 90)
    print("CONFLICTING LABEL EXAMPLES")
    print("=" * 90)

    conflict_texts = set(
        conflicting_messages.index
    )

    examples = combined[
        combined["normalized_text"]
        .isin(conflict_texts)
    ][
        [
            "source",
            "label",
            "text",
        ]
    ].head(20)

    print(
        examples.to_string(
            index=False
        )
    )


print("\n" + "=" * 90)
print("AUDIT COMPLETE")
print("=" * 90)