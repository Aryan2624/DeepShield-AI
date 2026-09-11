from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "phishing_urls"
    / "malicious_phish.csv"
)


def inspect_dataset():
    print("=" * 60)
    print("DeepShield AI - Malicious URL Dataset Inspection")
    print("=" * 60)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print("\n1. DATASET SHAPE")
    print("-" * 60)
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")

    print("\n2. COLUMN NAMES")
    print("-" * 60)
    print(df.columns.tolist())

    print("\n3. DATA TYPES")
    print("-" * 60)
    print(df.dtypes)

    print("\n4. MISSING VALUES")
    print("-" * 60)
    print(df.isnull().sum())

    print("\n5. DUPLICATE ROWS")
    print("-" * 60)
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    print("\n6. DUPLICATE URLs")
    print("-" * 60)
    print(f"Duplicate URLs: {df['url'].duplicated().sum():,}")

    print("\n7. CLASS DISTRIBUTION")
    print("-" * 60)

    class_counts = df["type"].value_counts()

    print(class_counts)

    print("\n8. CLASS PERCENTAGES")
    print("-" * 60)

    class_percentages = (
        df["type"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print(class_percentages)

    print("\n9. UNIQUE CLASSES")
    print("-" * 60)
    print(df["type"].unique().tolist())

    print("\n10. SAMPLE DATA")
    print("-" * 60)
    print(df.head())

    print("\n11. SAMPLE URL FROM EACH CLASS")
    print("-" * 60)

    for label in df["type"].dropna().unique():
        sample = df[df["type"] == label]["url"].iloc[0]

        print(f"\n{label.upper()}")
        print(sample)

    print("\n" + "=" * 60)
    print("Inspection complete.")
    print("=" * 60)


if __name__ == "__main__":
    inspect_dataset()