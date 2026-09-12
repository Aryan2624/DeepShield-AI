from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "phishing_urls"
    / "url_features_v2.csv"
)


def inspect_features():
    print("=" * 80)
    print("DeepShield AI - URL Feature Dataset v2 Inspection")
    print("=" * 80)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found:\n{DATA_PATH}"
        )

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("\n" + "=" * 80)
    print("DATASET SHAPE")
    print("=" * 80)

    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")

    print("\n" + "=" * 80)
    print("COLUMNS")
    print("=" * 80)

    for index, column in enumerate(
        df.columns,
        start=1,
    ):
        print(f"{index:02d}. {column}")

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("MISSING VALUES")
    print("=" * 80)

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values found.")
    else:
        print(missing.to_string())

    # ---------------------------------------------------------
    # Duplicate URLs
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("DUPLICATE URL CHECK")
    print("=" * 80)

    duplicate_urls = df["url"].duplicated().sum()

    print(
        f"Duplicate URLs : "
        f"{duplicate_urls:,}"
    )

    # ---------------------------------------------------------
    # Class distribution
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("CLASS DISTRIBUTION")
    print("=" * 80)

    class_counts = df["type"].value_counts()

    class_percentages = (
        df["type"]
        .value_counts(normalize=True)
        .mul(100)
    )

    distribution = pd.DataFrame({
        "count": class_counts,
        "percentage": class_percentages,
    })

    print(
        distribution.round(2).to_string()
    )

    # ---------------------------------------------------------
    # Numerical feature checks
    # ---------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column not in {"url", "type"}
    ]

    print("\n" + "=" * 80)
    print("FEATURE COUNT")
    print("=" * 80)

    print(
        f"Numerical ML features : "
        f"{len(feature_columns)}"
    )

    numeric_df = df[feature_columns]

    infinite_count = np.isinf(
        numeric_df.to_numpy(
            dtype=float
        )
    ).sum()

    print(
        f"Infinite values       : "
        f"{infinite_count:,}"
    )

    # ---------------------------------------------------------
    # Constant features
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("CONSTANT FEATURE CHECK")
    print("=" * 80)

    unique_counts = (
        numeric_df.nunique(
            dropna=False
        )
    )

    constant_features = (
        unique_counts[
            unique_counts <= 1
        ]
        .index
        .tolist()
    )

    if constant_features:
        print(
            "Constant features found:"
        )

        for feature in constant_features:
            print(f"- {feature}")

    else:
        print(
            "No constant features found."
        )

    # ---------------------------------------------------------
    # Feature summary
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("FEATURE SUMMARY")
    print("=" * 80)

    summary = (
        numeric_df
        .describe()
        .transpose()[
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        ]
    )

    print(
        summary.round(4).to_string()
    )

    print("\n" + "=" * 80)
    print("V2 INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    inspect_features()