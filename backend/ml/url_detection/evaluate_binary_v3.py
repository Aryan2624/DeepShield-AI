import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
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
    / "url_binary_v3_candidate.joblib"
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
    / "binary_benchmark_metrics.json"
)

THREAT_THRESHOLD = 0.45


def reconstruct_benchmark(features_df):
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

    return benchmark


def main():
    print("=" * 80)
    print(
        "DeepShield AI - "
        "Binary URL Threat Detector v3 Benchmark"
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

    print("\nLoading v3 binary model...")

    package = joblib.load(
        MODEL_PATH
    )

    model = package["model"]

    feature_columns = package[
        "feature_columns"
    ]

    print(
        f"Model            : "
        f"{package['model_name']}"
    )

    print(
        f"Feature engine   : "
        f"{package.get('feature_engine')}"
    )

    print(
        f"Threat threshold : "
        f"{THREAT_THRESHOLD:.2f}"
    )

    print(
        "\nLoading v2 feature dataset..."
    )

    features_df = pd.read_csv(
        FEATURES_PATH
    )

    benchmark = reconstruct_benchmark(
        features_df
    )

    print(
        f"Benchmark samples: "
        f"{len(benchmark):,}"
    )

    X = benchmark[
        feature_columns
    ]

    actual_threat = (
        benchmark["type"]
        != "benign"
    ).astype(int).to_numpy()

    print(
        "\nGenerating threat scores..."
    )

    probabilities = (
        model.predict_proba(X)
    )

    classes = list(
        model.classes_
    )

    threat_index = classes.index(1)

    threat_scores = probabilities[
        :,
        threat_index
    ]

    predictions = (
        threat_scores
        >= THREAT_THRESHOLD
    ).astype(int)

    accuracy = accuracy_score(
        actual_threat,
        predictions,
    )

    precision = precision_score(
        actual_threat,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        actual_threat,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        actual_threat,
        predictions,
        zero_division=0,
    )

    f2 = fbeta_score(
        actual_threat,
        predictions,
        beta=2,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        actual_threat,
        threat_scores,
    )

    pr_auc = average_precision_score(
        actual_threat,
        threat_scores,
    )

    true_positive = int(
        (
            (actual_threat == 1)
            & (predictions == 1)
        ).sum()
    )

    false_negative = int(
        (
            (actual_threat == 1)
            & (predictions == 0)
        ).sum()
    )

    false_positive = int(
        (
            (actual_threat == 0)
            & (predictions == 1)
        ).sum()
    )

    true_negative = int(
        (
            (actual_threat == 0)
            & (predictions == 0)
        ).sum()
    )

    false_alarm_rate = (
        false_positive
        /
        (
            false_positive
            + true_negative
        )
    )

    metrics = {
        "model_name":
            package["model_name"],
        "version":
            package.get("version"),
        "threshold":
            THREAT_THRESHOLD,
        "benchmark_samples":
            int(len(benchmark)),
        "accuracy":
            float(accuracy),
        "threat_precision":
            float(precision),
        "threat_recall":
            float(recall),
        "threat_f1":
            float(f1),
        "threat_f2":
            float(f2),
        "roc_auc":
            float(roc_auc),
        "pr_auc":
            float(pr_auc),
        "false_alarm_rate":
            float(false_alarm_rate),
        "true_positive":
            true_positive,
        "false_negative":
            false_negative,
        "false_positive":
            false_positive,
        "true_negative":
            true_negative,
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

    print("\n" + "=" * 80)
    print("V3 BINARY BENCHMARK METRICS")
    print("=" * 80)

    print(
        f"Accuracy               : "
        f"{accuracy:.4f}"
    )

    print(
        f"Threat Precision       : "
        f"{precision:.4f}"
    )

    print(
        f"Threat Recall          : "
        f"{recall:.4f}"
    )

    print(
        f"Threat F1              : "
        f"{f1:.4f}"
    )

    print(
        f"Threat F2              : "
        f"{f2:.4f}"
    )

    print(
        f"ROC-AUC                : "
        f"{roc_auc:.4f}"
    )

    print(
        f"PR-AUC                 : "
        f"{pr_auc:.4f}"
    )

    print(
        f"Benign False Alarm Rate: "
        f"{false_alarm_rate:.4f}"
    )

    print("\n" + "=" * 80)
    print("SECURITY COUNTS")
    print("=" * 80)

    print(
        f"Threats detected      : "
        f"{true_positive:,}"
    )

    print(
        f"Threats missed        : "
        f"{false_negative:,}"
    )

    print(
        f"Benign false alarms   : "
        f"{false_positive:,}"
    )

    print(
        f"Benign correctly safe : "
        f"{true_negative:,}"
    )

    print("\n" + "=" * 80)
    print("BINARY CONFUSION MATRIX")
    print("=" * 80)

    print(
        f"""
                     Predicted BENIGN    Predicted THREAT
Actual BENIGN        {true_negative:>16,}    {false_positive:>16,}
Actual THREAT        {false_negative:>16,}    {true_positive:>16,}
"""
    )

    print(
        f"Saved metrics:\n"
        f"{METRICS_PATH}"
    )

    print(
        "\nNOTE: Threshold 0.45 was selected "
        "using validation data only."
    )

    print(
        "This benchmark is an internal "
        "development benchmark."
    )


if __name__ == "__main__":
    main()