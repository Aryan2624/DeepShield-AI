from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "phishing_urls"
)

FEATURES_V2_PATH = DATA_DIR / "url_features_v2.csv"

TRAIN_V1_PATH = DATA_DIR / "url_train_v1.csv"
VALIDATION_V1_PATH = DATA_DIR / "url_validation_v1.csv"
TEST_V1_PATH = DATA_DIR / "url_test_v1.csv"

TRAIN_V2_PATH = DATA_DIR / "url_train_v2.csv"
VALIDATION_V2_PATH = DATA_DIR / "url_validation_v2.csv"
TEST_V2_PATH = DATA_DIR / "url_test_v2.csv"


def build_split(
    features_v2: pd.DataFrame,
    reference_path: Path,
    output_path: Path,
    split_name: str,
):
    print(f"\nBuilding {split_name} split...")

    reference = pd.read_csv(
        reference_path,
        usecols=["url", "type"],
    )

    reference_urls = set(reference["url"])

    split_df = features_v2[
        features_v2["url"].isin(reference_urls)
    ].copy()

    if len(split_df) != len(reference):
        raise ValueError(
            f"{split_name} row mismatch: "
            f"expected {len(reference):,}, "
            f"found {len(split_df):,}"
        )

    reference_labels = (
        reference
        .set_index("url")["type"]
        .to_dict()
    )

    mismatches = split_df[
        split_df.apply(
            lambda row:
                reference_labels.get(row["url"])
                != row["type"],
            axis=1,
        )
    ]

    if not mismatches.empty:
        raise ValueError(
            f"{split_name} contains label mismatches."
        )

    split_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"{split_name} rows : "
        f"{len(split_df):,}"
    )

    print(
        split_df["type"]
        .value_counts()
        .to_string()
    )

    return split_df


def main():
    print("=" * 75)
    print("DeepShield AI - URL Dataset v2 Split Reconstruction")
    print("=" * 75)

    required_files = [
        FEATURES_V2_PATH,
        TRAIN_V1_PATH,
        VALIDATION_V1_PATH,
        TEST_V1_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    print("\nLoading v2 feature dataset...")

    features_v2 = pd.read_csv(
        FEATURES_V2_PATH
    )

    print(
        f"Total v2 rows : "
        f"{len(features_v2):,}"
    )

    train_df = build_split(
        features_v2,
        TRAIN_V1_PATH,
        TRAIN_V2_PATH,
        "TRAIN",
    )

    validation_df = build_split(
        features_v2,
        VALIDATION_V1_PATH,
        VALIDATION_V2_PATH,
        "VALIDATION",
    )

    test_df = build_split(
        features_v2,
        TEST_V1_PATH,
        TEST_V2_PATH,
        "BENCHMARK TEST",
    )

    print("\n" + "=" * 75)
    print("LEAKAGE CHECK")
    print("=" * 75)

    train_urls = set(train_df["url"])
    validation_urls = set(
        validation_df["url"]
    )
    test_urls = set(test_df["url"])

    print(
        "Train ↔ Validation overlap :",
        len(
            train_urls
            & validation_urls
        ),
    )

    print(
        "Train ↔ Test overlap       :",
        len(
            train_urls
            & test_urls
        ),
    )

    print(
        "Validation ↔ Test overlap  :",
        len(
            validation_urls
            & test_urls
        ),
    )

    total_rows = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    print("\n" + "=" * 75)
    print("V2 SPLIT SUMMARY")
    print("=" * 75)

    print(
        f"Training   : "
        f"{len(train_df):,}"
    )

    print(
        f"Validation : "
        f"{len(validation_df):,}"
    )

    print(
        f"Benchmark  : "
        f"{len(test_df):,}"
    )

    print(
        f"Total      : "
        f"{total_rows:,}"
    )

    if total_rows != len(features_v2):
        raise ValueError(
            "Split total does not match "
            "the v2 feature dataset."
        )

    print("\nSaved:")
    print(TRAIN_V2_PATH)
    print(VALIDATION_V2_PATH)
    print(TEST_V2_PATH)


if __name__ == "__main__":
    main()