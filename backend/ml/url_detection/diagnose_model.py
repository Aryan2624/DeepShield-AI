from pathlib import Path

import joblib
import pandas as pd

try:
    from .features import extract_url_features
except ImportError:
    from features import extract_url_features


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "saved_models"
    / "url_detector_v1.joblib"
)

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "phishing_urls"
)

TRAIN_PATH = DATASET_DIR / "url_train.csv"
VALIDATION_PATH = DATASET_DIR / "url_validation.csv"
TEST_PATH = DATASET_DIR / "url_test.csv"


TEST_URLS = [
    "google.com",
    "www.google.com",
    "http://www.google.com",
    "https://www.google.com",
]


def main():
    print("=" * 80)
    print("DeepShield AI - URL Model Diagnostic")
    print("=" * 80)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    package = joblib.load(MODEL_PATH)

    model = package["model"]
    feature_columns = package["feature_columns"]

    print(f"\nModel          : {package['model_name']}")
    print(f"Feature count  : {len(feature_columns)}")
    print(f"Classes        : {list(model.classes_)}")

    # ---------------------------------------------------------
    # 1. Feature importance
    # ---------------------------------------------------------

    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            "This model does not expose feature_importances_."
        )

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("TOP 15 MODEL FEATURES")
    print("=" * 80)

    print(
        importance_df.head(15).to_string(
            index=False
        )
    )

    top_features = (
        importance_df
        .head(10)["feature"]
        .tolist()
    )

    # ---------------------------------------------------------
    # 2. Google URL feature values
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("GOOGLE URL FEATURE VALUES")
    print("=" * 80)

    rows = []

    for url in TEST_URLS:
        features = extract_url_features(url)

        row = {
            "url": url,
        }

        for feature in top_features:
            row[feature] = features[feature]

        rows.append(row)

    google_feature_df = pd.DataFrame(rows)

    print(
        google_feature_df.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # 3. Check whether Google appears in datasets
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("GOOGLE.COM OCCURRENCES IN DATASET")
    print("=" * 80)

    split_paths = {
        "TRAIN": TRAIN_PATH,
        "VALIDATION": VALIDATION_PATH,
        "TEST": TEST_PATH,
    }

    normalized_targets = {
        url.lower()
        for url in TEST_URLS
    }

    for split_name, path in split_paths.items():

        df = pd.read_csv(
            path,
            usecols=["url", "type"],
        )

        url_normalized = (
            df["url"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        exact_mask = url_normalized.isin(
            normalized_targets
        )

        contains_mask = url_normalized.str.contains(
            "google.com",
            regex=False,
            na=False,
        )

        exact_matches = df.loc[
            exact_mask,
            ["url", "type"],
        ]

        google_matches = df.loc[
            contains_mask,
            ["url", "type"],
        ]

        print(f"\n{split_name}")

        print(
            f"Exact Google variants : "
            f"{len(exact_matches):,}"
        )

        print(
            f"URLs containing google.com : "
            f"{len(google_matches):,}"
        )

        if len(google_matches) > 0:

            print("\nLabel counts:")

            print(
                google_matches["type"]
                .value_counts()
                .to_string()
            )

            print("\nSample rows:")

            print(
                google_matches
                .head(10)
                .to_string(index=False)
            )

    # ---------------------------------------------------------
    # 4. Compare important features by class
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("TRAINING CLASS FEATURE AVERAGES")
    print("=" * 80)

    train_columns = [
        "type",
        *top_features,
    ]

    train_df = pd.read_csv(
        TRAIN_PATH,
        usecols=train_columns,
    )

    class_means = (
        train_df
        .groupby("type")[top_features]
        .mean()
        .transpose()
    )

    print(
        class_means.round(4).to_string()
    )

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()