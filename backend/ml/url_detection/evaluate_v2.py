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

FEATURES_V2_PATH = DATA_DIR / "url_features_v2.csv"

BENCHMARK_REFERENCE_PATH = (
    DATA_DIR / "url_test_v1.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "saved_models"
    / "url_detector_v2_candidate.joblib"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "url_detection"
    / "evaluation_v2"
)

METRICS_PATH = (
    RESULT_DIR
    / "benchmark_metrics.json"
)

REPORT_PATH = (
    RESULT_DIR
    / "classification_report.csv"
)

CONFUSION_PATH = (
    RESULT_DIR
    / "confusion_matrix.csv"
)


def reconstruct_benchmark():
    print(
        "\nLoading complete v2 feature dataset..."
    )

    features_v2 = pd.read_csv(
        FEATURES_V2_PATH
    )

    print(
        f"Total feature rows : "
        f"{len(features_v2):,}"
    )

    print(
        "\nLoading benchmark membership..."
    )

    reference = pd.read_csv(
        BENCHMARK_REFERENCE_PATH,
        usecols=["url", "type"],
    )

    benchmark_urls = set(
        reference["url"]
    )

    benchmark = features_v2[
        features_v2["url"].isin(
            benchmark_urls
        )
    ].copy()

    if len(benchmark) != len(reference):
        raise ValueError(
            "Benchmark reconstruction failed: "
            f"expected {len(reference):,}, "
            f"found {len(benchmark):,}"
        )

    reference_labels = (
        reference
        .set_index("url")["type"]
        .to_dict()
    )

    mismatches = benchmark[
        benchmark.apply(
            lambda row:
                reference_labels.get(
                    row["url"]
                )
                != row["type"],
            axis=1,
        )
    ]

    if not mismatches.empty:
        raise ValueError(
            "Benchmark contains label mismatches."
        )

    print(
        f"Benchmark samples  : "
        f"{len(benchmark):,}"
    )

    return benchmark


def main():
    print("=" * 80)
    print(
        "DeepShield AI - "
        "URL Detector v2 Internal Benchmark"
    )
    print("=" * 80)

    required_files = [
        FEATURES_V2_PATH,
        BENCHMARK_REFERENCE_PATH,
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

    print("\nLoading v2 candidate model...")

    package = joblib.load(
        MODEL_PATH
    )

    model = package["model"]

    model_name = package[
        "model_name"
    ]

    feature_columns = package[
        "feature_columns"
    ]

    print(
        f"Model          : {model_name}"
    )

    print(
        f"Feature engine : "
        f"{package.get('feature_engine', 'unknown')}"
    )

    print(
        f"Features       : "
        f"{len(feature_columns)}"
    )

    benchmark = (
        reconstruct_benchmark()
    )

    missing_features = [
        feature
        for feature in feature_columns
        if feature
        not in benchmark.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing benchmark features: "
            + ", ".join(
                missing_features
            )
        )

    X_benchmark = benchmark[
        feature_columns
    ]

    y_benchmark = benchmark[
        "type"
    ]

    print(
        "\nGenerating benchmark predictions..."
    )

    predictions = model.predict(
        X_benchmark
    )

    probabilities = (
        model.predict_proba(
            X_benchmark
        )
    )

    # ---------------------------------------------------------
    # Overall multiclass metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_benchmark,
        predictions,
    )

    precision_macro = precision_score(
        y_benchmark,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        y_benchmark,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_benchmark,
        predictions,
        average="macro",
        zero_division=0,
    )

    precision_weighted = precision_score(
        y_benchmark,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall_weighted = recall_score(
        y_benchmark,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_benchmark,
        predictions,
        average="weighted",
        zero_division=0,
    )

    roc_auc_macro = roc_auc_score(
        y_benchmark,
        probabilities,
        labels=model.classes_,
        multi_class="ovr",
        average="macro",
    )

    # ---------------------------------------------------------
    # Per-class report
    # ---------------------------------------------------------

    report_dict = classification_report(
        y_benchmark,
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

    # ---------------------------------------------------------
    # Multiclass confusion matrix
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y_benchmark,
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

    # ---------------------------------------------------------
    # Security-level binary view
    #
    # benign     -> SAFE
    # everything else -> THREAT
    # ---------------------------------------------------------

    actual_threat = (
        y_benchmark != "benign"
    )

    predicted_threat = (
        predictions != "benign"
    )

    threat_tp = int(
        (
            actual_threat
            & predicted_threat
        ).sum()
    )

    threat_fn = int(
        (
            actual_threat
            & ~predicted_threat
        ).sum()
    )

    false_alarm_count = int(
        (
            ~actual_threat
            & predicted_threat
        ).sum()
    )

    true_benign_count = int(
        (
            ~actual_threat
            & ~predicted_threat
        ).sum()
    )

    threat_detection_recall = (
        threat_tp
        /
        (threat_tp + threat_fn)
        if (
            threat_tp
            + threat_fn
        ) > 0
        else 0.0
    )

    benign_false_alarm_rate = (
        false_alarm_count
        /
        (
            false_alarm_count
            + true_benign_count
        )
        if (
            false_alarm_count
            + true_benign_count
        ) > 0
        else 0.0
    )

    metrics = {
        "model_name": model_name,
        "benchmark_samples":
            int(len(X_benchmark)),
        "feature_count":
            int(len(feature_columns)),
        "accuracy":
            float(accuracy),
        "precision_macro":
            float(precision_macro),
        "recall_macro":
            float(recall_macro),
        "f1_macro":
            float(f1_macro),
        "precision_weighted":
            float(precision_weighted),
        "recall_weighted":
            float(recall_weighted),
        "f1_weighted":
            float(f1_weighted),
        "roc_auc_macro":
            float(roc_auc_macro),
        "threat_detection_recall":
            float(
                threat_detection_recall
            ),
        "benign_false_alarm_rate":
            float(
                benign_false_alarm_rate
            ),
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

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("V2 INTERNAL BENCHMARK METRICS")
    print("=" * 80)

    print(
        f"Accuracy               : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision Macro        : "
        f"{precision_macro:.4f}"
    )

    print(
        f"Recall Macro           : "
        f"{recall_macro:.4f}"
    )

    print(
        f"F1 Macro               : "
        f"{f1_macro:.4f}"
    )

    print(
        f"F1 Weighted            : "
        f"{f1_weighted:.4f}"
    )

    print(
        f"ROC-AUC Macro          : "
        f"{roc_auc_macro:.4f}"
    )

    print(
        f"Threat Detection Recall: "
        f"{threat_detection_recall:.4f}"
    )

    print(
        f"Benign False Alarm Rate: "
        f"{benign_false_alarm_rate:.4f}"
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
    print("CONFUSION MATRIX")
    print("=" * 80)

    print(
        confusion_df.to_string()
    )

    print("\n" + "=" * 80)
    print("SECURITY VIEW")
    print("=" * 80)

    print(
        f"Threats detected     : "
        f"{threat_tp:,}"
    )

    print(
        f"Threats missed       : "
        f"{threat_fn:,}"
    )

    print(
        f"Benign false alarms  : "
        f"{false_alarm_count:,}"
    )

    print(
        f"Benign correctly safe: "
        f"{true_benign_count:,}"
    )

    print("\n" + "=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(METRICS_PATH)
    print(REPORT_PATH)
    print(CONFUSION_PATH)

    print(
        "\nNOTE: This is an internal benchmark, "
        "not a pristine untouched final test."
    )


if __name__ == "__main__":
    main()