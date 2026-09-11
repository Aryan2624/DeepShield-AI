from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "phishing_urls"
    / "url_features.csv"
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "phishing_urls"
    / "url_train.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "phishing_urls"
    / "url_validation.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "phishing_urls"
    / "url_test.csv"
)


def show_distribution(name, dataframe):
    print(f"\n{name}")
    print("-" * 65)

    print(
        dataframe["type"]
        .value_counts()
    )

    print("\nPercentages:")

    print(
        dataframe["type"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )


def split_dataset():
    print("=" * 65)
    print("DeepShield AI - Train / Validation / Test Split")
    print("=" * 65)

    if not FEATURE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found:\n{FEATURE_DATA_PATH}"
        )

    df = pd.read_csv(
        FEATURE_DATA_PATH
    )

    print(
        f"\nTotal rows: "
        f"{len(df):,}"
    )


    # -------------------------------------------------
    # First split
    #
    # 70% training
    # 30% temporary
    # -------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["type"],
    )


    # -------------------------------------------------
    # Second split
    #
    # Temporary 30% becomes:
    #
    # 15% validation
    # 15% final test
    # -------------------------------------------------

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["type"],
    )


    # -------------------------------------------------
    # Reset indices
    # -------------------------------------------------

    train_df = (
        train_df
        .reset_index(drop=True)
    )

    validation_df = (
        validation_df
        .reset_index(drop=True)
    )

    test_df = (
        test_df
        .reset_index(drop=True)
    )


    # -------------------------------------------------
    # Check for URL leakage between splits
    # -------------------------------------------------

    train_urls = set(
        train_df["url"]
    )

    validation_urls = set(
        validation_df["url"]
    )

    test_urls = set(
        test_df["url"]
    )


    train_validation_overlap = len(
        train_urls
        & validation_urls
    )

    train_test_overlap = len(
        train_urls
        & test_urls
    )

    validation_test_overlap = len(
        validation_urls
        & test_urls
    )


    print("\n" + "=" * 65)
    print("SPLIT SIZES")
    print("=" * 65)

    print(
        f"Training   : "
        f"{len(train_df):,}"
    )

    print(
        f"Validation : "
        f"{len(validation_df):,}"
    )

    print(
        f"Test       : "
        f"{len(test_df):,}"
    )


    print("\n" + "=" * 65)
    print("URL LEAKAGE CHECK")
    print("=" * 65)

    print(
        "Train ↔ Validation overlap : "
        f"{train_validation_overlap}"
    )

    print(
        "Train ↔ Test overlap       : "
        f"{train_test_overlap}"
    )

    print(
        "Validation ↔ Test overlap  : "
        f"{validation_test_overlap}"
    )


    if (
        train_validation_overlap != 0
        or train_test_overlap != 0
        or validation_test_overlap != 0
    ):
        raise ValueError(
            "URL leakage detected between dataset splits."
        )


    show_distribution(
        "TRAINING DISTRIBUTION",
        train_df,
    )

    show_distribution(
        "VALIDATION DISTRIBUTION",
        validation_df,
    )

    show_distribution(
        "TEST DISTRIBUTION",
        test_df,
    )


    # -------------------------------------------------
    # Save datasets
    # -------------------------------------------------

    train_df.to_csv(
        TRAIN_PATH,
        index=False,
    )

    validation_df.to_csv(
        VALIDATION_PATH,
        index=False,
    )

    test_df.to_csv(
        TEST_PATH,
        index=False,
    )


    print("\n" + "=" * 65)
    print("SPLIT COMPLETE")
    print("=" * 65)

    print(
        f"\nTraining set saved to:\n"
        f"{TRAIN_PATH}"
    )

    print(
        f"\nValidation set saved to:\n"
        f"{VALIDATION_PATH}"
    )

    print(
        f"\nTest set saved to:\n"
        f"{TEST_PATH}"
    )


if __name__ == "__main__":
    split_dataset()