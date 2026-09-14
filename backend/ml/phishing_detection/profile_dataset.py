from pathlib import Path

import pandas as pd


DATASET_DIR = Path(r"E:\DeepShield-Datasets\phishing_messages")


FILES = [
    "CEAS_08.csv",
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nigerian_Fraud.csv",
    "SpamAssasin.csv",
    "phishing_email.csv",
]


def build_text(df, filename):
    if filename == "phishing_email.csv":
        text = df["text_combined"].fillna("").astype(str)

    else:
        subject = df["subject"].fillna("").astype(str)
        body = df["body"].fillna("").astype(str)

        text = (
            subject.str.strip()
            + " "
            + body.str.strip()
        ).str.strip()

    return text


print("=" * 90)
print("DEEPSHIELD AI - PHISHING DATASET PROFILE")
print("=" * 90)


for filename in FILES:
    path = DATASET_DIR / filename

    print("\n" + "=" * 90)
    print(f"FILE: {filename}")
    print("=" * 90)

    if filename == "phishing_email.csv":
        df = pd.read_csv(
            path,
            usecols=["text_combined", "label"],
            low_memory=False,
        )

    else:
        df = pd.read_csv(
            path,
            usecols=["subject", "body", "label"],
            low_memory=False,
        )

    text = build_text(df, filename)

    print(f"Rows: {len(df):,}")

    print("\nLabel distribution:")
    print(
        df["label"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nLabel percentages:")
    percentages = (
        df["label"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print(percentages)

    empty_text = text.str.strip().eq("").sum()

    print(f"\nEmpty messages: {empty_text:,}")

    normalized_text = (
        text
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    duplicate_count = normalized_text.duplicated().sum()

    print(f"Duplicate messages inside file: {duplicate_count:,}")

    print(
        f"Unique normalized messages: "
        f"{normalized_text.nunique():,}"
    )


print("\n" + "=" * 90)
print("PROFILE COMPLETE")
print("=" * 90)