from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "phishing_urls"
    / "malicious_phish.csv"
)

CLEAN_DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "phishing_urls"
    / "malicious_phish_clean.csv"
)


VALID_CLASSES = {
    "benign",
    "phishing",
    "defacement",
    "malware",
}


def clean_dataset():
    print("=" * 65)
    print("DeepShield AI - URL Dataset Cleaning")
    print("=" * 65)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    original_rows = len(df)

    print(f"\nOriginal rows: {original_rows:,}")


    # -------------------------------------------------
    # 1. Keep only required columns
    # -------------------------------------------------

    df = df[["url", "type"]].copy()


    # -------------------------------------------------
    # 2. Remove missing values
    # -------------------------------------------------

    missing_before = df.isnull().sum().sum()

    df = df.dropna(subset=["url", "type"])

    print(
        f"Rows with missing required values removed: "
        f"{missing_before:,}"
    )


    # -------------------------------------------------
    # 3. Clean surrounding whitespace
    # -------------------------------------------------

    df["url"] = df["url"].astype(str).str.strip()

    df["type"] = (
        df["type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )


    # -------------------------------------------------
    # 4. Remove empty values
    # -------------------------------------------------

    empty_rows = (
        (df["url"] == "")
        | (df["type"] == "")
    ).sum()

    df = df[
        (df["url"] != "")
        & (df["type"] != "")
    ]

    print(f"Empty rows removed: {empty_rows:,}")


    # -------------------------------------------------
    # 5. Validate labels
    # -------------------------------------------------

    invalid_labels = ~df["type"].isin(VALID_CLASSES)

    invalid_count = invalid_labels.sum()

    if invalid_count > 0:
        print(
            f"Invalid-label rows removed: "
            f"{invalid_count:,}"
        )

        df = df[~invalid_labels]
    else:
        print("Invalid-label rows removed: 0")


    # -------------------------------------------------
    # 6. Remove exact duplicate rows
    # -------------------------------------------------

    duplicate_rows = df.duplicated().sum()

    df = df.drop_duplicates()

    print(
        f"Exact duplicate rows removed: "
        f"{duplicate_rows:,}"
    )


    # -------------------------------------------------
    # 7. Detect URLs with conflicting labels
    # -------------------------------------------------

    label_counts = (
        df.groupby("url")["type"]
        .nunique()
    )

    conflicting_urls = label_counts[
        label_counts > 1
    ].index

    conflict_count = len(conflicting_urls)

    print(
        f"URLs with conflicting labels: "
        f"{conflict_count:,}"
    )


    # Show examples if conflicts exist
    if conflict_count > 0:

        print("\nExample conflicting URLs:")

        conflicts = (
            df[df["url"].isin(conflicting_urls)]
            .sort_values("url")
        )

        print(
            conflicts.head(20).to_string(
                index=False
            )
        )


    # -------------------------------------------------
    # 8. Remove ambiguous URLs
    # -------------------------------------------------

    conflict_rows = df[
        df["url"].isin(conflicting_urls)
    ].shape[0]

    df = df[
        ~df["url"].isin(conflicting_urls)
    ]

    print(
        f"Rows removed because of conflicting labels: "
        f"{conflict_rows:,}"
    )


    # -------------------------------------------------
    # 9. Final URL uniqueness check
    # -------------------------------------------------

    remaining_duplicate_urls = (
        df["url"]
        .duplicated()
        .sum()
    )

    print(
        f"Remaining duplicate URLs: "
        f"{remaining_duplicate_urls:,}"
    )


    # -------------------------------------------------
    # 10. Shuffle dataset reproducibly
    # -------------------------------------------------

    df = (
        df.sample(
            frac=1,
            random_state=42,
        )
        .reset_index(drop=True)
    )


    # -------------------------------------------------
    # 11. Save cleaned dataset
    # -------------------------------------------------

    df.to_csv(
        CLEAN_DATA_PATH,
        index=False,
    )


    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    final_rows = len(df)

    removed_rows = (
        original_rows
        - final_rows
    )

    print("\n" + "=" * 65)
    print("CLEANING SUMMARY")
    print("=" * 65)

    print(
        f"Original rows : "
        f"{original_rows:,}"
    )

    print(
        f"Final rows    : "
        f"{final_rows:,}"
    )

    print(
        f"Total removed : "
        f"{removed_rows:,}"
    )


    print("\nFinal class distribution:")

    print(
        df["type"]
        .value_counts()
    )


    print("\nFinal class percentages:")

    percentages = (
        df["type"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print(percentages)


    print(
        f"\nClean dataset saved to:\n"
        f"{CLEAN_DATA_PATH}"
    )

    print("\nCleaning complete.")


if __name__ == "__main__":
    clean_dataset()