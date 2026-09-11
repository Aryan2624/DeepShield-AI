from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "phishing_urls"
    / "url_features.csv"
)


def inspect_feature_dataset():
    print("=" * 70)
    print("DeepShield AI - URL Feature Dataset Inspection")
    print("=" * 70)

    if not FEATURE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found:\n{FEATURE_DATA_PATH}"
        )

    df = pd.read_csv(FEATURE_DATA_PATH)

    print("\n1. DATASET SHAPE")
    print("-" * 70)
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")

    print("\n2. COLUMN NAMES")
    print("-" * 70)

    for index, column in enumerate(df.columns, start=1):
        print(f"{index:>2}. {column}")

    print("\n3. MISSING VALUES")
    print("-" * 70)

    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if missing_values.empty:
        print("No missing values found.")
    else:
        print(missing_values)

    print("\n4. INFINITE VALUES")
    print("-" * 70)

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    infinite_count = (
        np.isinf(df[numeric_columns])
        .sum()
        .sum()
    )

    print(
        f"Infinite numeric values: "
        f"{infinite_count:,}"
    )

    print("\n5. DUPLICATE URLS")
    print("-" * 70)

    duplicate_urls = (
        df["url"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate URLs: "
        f"{duplicate_urls:,}"
    )

    print("\n6. CLASS DISTRIBUTION")
    print("-" * 70)

    print(
        df["type"]
        .value_counts()
    )

    print("\n7. CLASS PERCENTAGES")
    print("-" * 70)

    print(
        df["type"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    feature_columns = [
        column
        for column in df.columns
        if column not in {
            "url",
            "type",
        }
    ]

    print("\n8. FEATURE COUNT")
    print("-" * 70)

    print(
        f"Numerical ML features: "
        f"{len(feature_columns)}"
    )

    print("\n9. IMPORTANT FEATURE CHECKS")
    print("-" * 70)

    important_features = [
        "has_ip_address",
        "shortened_url",
        "url_parse_error",
        "non_ascii_count",
        "subdomain_count",
        "url_entropy",
    ]

    for feature in important_features:
        if feature in df.columns:
            print(f"\n{feature}")
            print(
                df[feature]
                .describe()
                .round(4)
            )

    print("\n10. FEATURE SUMMARY")
    print("-" * 70)

    print(
        df[feature_columns]
        .describe()
        .T[
            [
                "min",
                "mean",
                "std",
                "max",
            ]
        ]
        .round(4)
    )

    print("\n" + "=" * 70)
    print("Feature dataset inspection complete.")
    print("=" * 70)


if __name__ == "__main__":
    inspect_feature_dataset()