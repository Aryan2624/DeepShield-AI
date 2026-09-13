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
    / "url_detector_v2_candidate.joblib"
)

THREAT_THRESHOLD = 0.53


def reconstruct_benchmark(
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

    return benchmark


def two_stage_predictions(
    model,
    probabilities,
):
    classes = list(
        model.classes_
    )

    benign_index = classes.index(
        "benign"
    )

    threat_classes = [
        label
        for label in classes
        if label != "benign"
    ]

    predictions = []

    threat_scores = []

    for probability_row in probabilities:

        benign_probability = (
            probability_row[
                benign_index
            ]
        )

        threat_score = (
            1.0
            - benign_probability
        )

        threat_scores.append(
            threat_score
        )

        if (
            threat_score
            < THREAT_THRESHOLD
        ):
            predictions.append(
                "benign"
            )

            continue

        best_threat_class = max(
            threat_classes,
            key=lambda label:
                probability_row[
                    classes.index(
                        label
                    )
                ],
        )

        predictions.append(
            best_threat_class
        )

    return (
        predictions,
        threat_scores,
    )


def main():
    print("=" * 80)
    print(
        "DeepShield AI - "
        "v2 Tuned Threshold Benchmark"
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
                f"Required file not found:\n"
                f"{path}"
            )

    print(
        "\nLoading model..."
    )

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
        f"Threat threshold : "
        f"{THREAT_THRESHOLD:.2f}"
    )

    print(
        "\nLoading v2 features..."
    )

    features_df = pd.read_csv(
        FEATURES_PATH
    )

    benchmark = (
        reconstruct_benchmark(
            features_df
        )
    )

    print(
        f"Benchmark samples: "
        f"{len(benchmark):,}"
    )

    X = benchmark[
        feature_columns
    ]

    y = benchmark[
        "type"
    ]

    print(
        "\nGenerating probabilities..."
    )

    probabilities = (
        model.predict_proba(X)
    )

    baseline_predictions = (
        model.predict(X)
    )

    (
        tuned_predictions,
        threat_scores,
    ) = two_stage_predictions(
        model,
        probabilities,
    )

    # -----------------------------------------------------
    # BASELINE SECURITY VIEW
    # -----------------------------------------------------

    actual_threat = (
        y != "benign"
    ).to_numpy()

    baseline_threat = (
        baseline_predictions
        != "benign"
    )

    baseline_recall = recall_score(
        actual_threat,
        baseline_threat,
        zero_division=0,
    )

    baseline_precision = precision_score(
        actual_threat,
        baseline_threat,
        zero_division=0,
    )

    baseline_false_alarm = (
        (
            (~actual_threat)
            & baseline_threat
        ).sum()
        /
        (~actual_threat).sum()
    )

    # -----------------------------------------------------
    # TUNED SECURITY VIEW
    # -----------------------------------------------------

    tuned_threat = (
        pd.Series(
            tuned_predictions
        )
        != "benign"
    ).to_numpy()

    threat_precision = precision_score(
        actual_threat,
        tuned_threat,
        zero_division=0,
    )

    threat_recall = recall_score(
        actual_threat,
        tuned_threat,
        zero_division=0,
    )

    threat_f1 = f1_score(
        actual_threat,
        tuned_threat,
        zero_division=0,
    )

    true_positive = int(
        (
            actual_threat
            & tuned_threat
        ).sum()
    )

    false_negative = int(
        (
            actual_threat
            & (~tuned_threat)
        ).sum()
    )

    false_positive = int(
        (
            (~actual_threat)
            & tuned_threat
        ).sum()
    )

    true_negative = int(
        (
            (~actual_threat)
            & (~tuned_threat)
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

    # -----------------------------------------------------
    # MULTICLASS METRICS
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y,
        tuned_predictions,
    )

    macro_f1 = f1_score(
        y,
        tuned_predictions,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y,
        tuned_predictions,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    classes = list(
        model.classes_
    )

    cm = confusion_matrix(
        y,
        tuned_predictions,
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

    # -----------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BASELINE VS TUNED SECURITY VIEW"
    )

    print(
        "=" * 80
    )

    print(
        "\nBaseline Random Forest:"
    )

    print(
        f"Threat Precision       : "
        f"{baseline_precision:.4f}"
    )

    print(
        f"Threat Recall          : "
        f"{baseline_recall:.4f}"
    )

    print(
        f"Benign False Alarm Rate: "
        f"{baseline_false_alarm:.4f}"
    )

    print(
        "\nTuned Two-Stage Detector:"
    )

    print(
        f"Threshold              : "
        f"{THREAT_THRESHOLD:.2f}"
    )

    print(
        f"Threat Precision       : "
        f"{threat_precision:.4f}"
    )

    print(
        f"Threat Recall          : "
        f"{threat_recall:.4f}"
    )

    print(
        f"Threat F1              : "
        f"{threat_f1:.4f}"
    )

    print(
        f"Benign False Alarm Rate: "
        f"{false_alarm_rate:.4f}"
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TUNED MULTICLASS PERFORMANCE"
    )

    print(
        "=" * 80
    )

    print(
        f"Accuracy : "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro F1 : "
        f"{macro_f1:.4f}"
    )

    print(
        "\n"
        + report_df.loc[
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

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TUNED CONFUSION MATRIX"
    )

    print(
        "=" * 80
    )

    print(
        confusion_df.to_string()
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SECURITY COUNTS"
    )

    print(
        "=" * 80
    )

    print(
        f"Threats detected     : "
        f"{true_positive:,}"
    )

    print(
        f"Threats missed       : "
        f"{false_negative:,}"
    )

    print(
        f"Benign false alarms  : "
        f"{false_positive:,}"
    )

    print(
        f"Benign correctly safe: "
        f"{true_negative:,}"
    )

    print(
        "\nThreshold was selected "
        "using validation data, "
        "not this benchmark."
    )


if __name__ == "__main__":
    main()