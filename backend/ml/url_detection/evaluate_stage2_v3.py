import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "phishing_urls"
)

FEATURES_PATH = (
    DATA_DIR
    / "url_features_v2.csv"
)

BENCHMARK_REFERENCE = (
    DATA_DIR
    / "url_test_v1.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "saved_models"
    / "url_threat_type_v3_candidate.joblib"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "url_detection"
    / "evaluation_v3"
)

METRICS_PATH = (
    RESULT_DIR
    / "stage2_benchmark_metrics.json"
)

REPORT_PATH = (
    RESULT_DIR
    / "stage2_classification_report.csv"
)

CONFUSION_PATH = (
    RESULT_DIR
    / "stage2_confusion_matrix.csv"
)


def reconstruct_threat_benchmark(
    features_df,
):
    reference = pd.read_csv(
        BENCHMARK_REFERENCE,
        usecols=["url", "type"],
    )

    benchmark_urls = set(
        reference["url"]
    )

    benchmark = features_df[
        features_df["url"].isin(
            benchmark_urls
        )
    ].copy()

    if len(benchmark) != len(reference):
        raise ValueError(
            "Benchmark reconstruction mismatch: "
            f"expected {len(reference):,}, "
            f"found {len(benchmark):,}"
        )

    threat_benchmark = benchmark[
        benchmark["type"] != "benign"
    ].copy()

    print(
        f"All benchmark samples   : "
        f"{len(benchmark):,}"
    )

    print(
        f"Threat benchmark samples: "
        f"{len(threat_benchmark):,}"
    )

    print(
        "\nThreat class distribution:"
    )

    print(
        threat_benchmark["type"]
        .value_counts()
        .to_string()
    )

    return threat_benchmark


def main():
    print("=" * 80)
    print(
        "DeepShield AI - "
        "Stage 2 Threat Classifier v3 Benchmark"
    )
    print("=" * 80)

    required_files = [
        FEATURES_PATH,
        BENCHMARK_REFERENCE,
        MODEL_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLoading Stage 2 candidate...")

    package = joblib.load(
        MODEL_PATH
    )

    model = package["model"]

    feature_columns = package[
        "feature_columns"
    ]

    print(
        f"Model          : "
        f"{package['model_name']}"
    )

    print(
        f"Feature engine : "
        f"{package.get('feature_engine')}"
    )

    print(
        f"Features       : "
        f"{len(feature_columns)}"
    )

    print(
        "\nLoading feature dataset..."
    )

    features_df = pd.read_csv(
        FEATURES_PATH
    )

    benchmark = reconstruct_threat_benchmark(
        features_df
    )

    X = benchmark[
        feature_columns
    ]

    y = benchmark[
        "type"
    ]

    print(
        "\nGenerating Stage 2 predictions..."
    )

    predictions = model.predict(X)

    probabilities = (
        model.predict_proba(X)
    )

    accuracy = accuracy_score(
        y,
        predictions,
    )

    precision_macro = precision_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0,
    )

    roc_auc_macro = roc_auc_score(
        y,
        probabilities,
        labels=model.classes_,
        multi_class="ovr",
        average="macro",
    )

    report_dict = classification_report(
        y,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report_dict
    ).transpose()

    classes = list(
        model.classes_
    )

    cm = confusion_matrix(
        y,
        predictions,
        labels=classes,
    )

    confusion_df = pd.DataFrame(
        cm,
        index=[
            f"Actual_{label}"
            for label in classes
        ],
        columns=[
            f"Predicted_{label}"
            for label in classes
        ],
    )

    metrics = {
        "model_name":
            package["model_name"],
        "benchmark_samples":
            int(len(benchmark)),
        "accuracy":
            float(accuracy),
        "precision_macro":
            float(precision_macro),
        "recall_macro":
            float(recall_macro),
        "f1_macro":
            float(f1_macro),
        "f1_weighted":
            float(f1_weighted),
        "roc_auc_macro":
            float(roc_auc_macro),
        "classes":
            classes,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    report_df.to_csv(
        REPORT_PATH
    )

    confusion_df.to_csv(
        CONFUSION_PATH
    )

    print("\n" + "=" * 80)
    print("V3 STAGE 2 BENCHMARK METRICS")
    print("=" * 80)

    print(
        f"Accuracy        : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision Macro : "
        f"{precision_macro:.4f}"
    )

    print(
        f"Recall Macro    : "
        f"{recall_macro:.4f}"
    )

    print(
        f"F1 Macro        : "
        f"{f1_macro:.4f}"
    )

    print(
        f"F1 Weighted     : "
        f"{f1_weighted:.4f}"
    )

    print(
        f"ROC-AUC Macro   : "
        f"{roc_auc_macro:.4f}"
    )

    print("\n" + "=" * 80)
    print("PER-CLASS PERFORMANCE")
    print("=" * 80)

    print(
        report_df.loc[
            classes,
            [
                "precision",
                "recall",
                "f1-score",
                "support",
            ],
        ]
        .round(4)
        .to_string()
    )

    print("\n" + "=" * 80)
    print("STAGE 2 CONFUSION MATRIX")
    print("=" * 80)

    print(
        confusion_df.to_string()
    )

    print("\nSaved:")
    print(METRICS_PATH)
    print(REPORT_PATH)
    print(CONFUSION_PATH)

    print(
        "\nNOTE: This is an internal "
        "development benchmark."
    )


if __name__ == "__main__":
    main()