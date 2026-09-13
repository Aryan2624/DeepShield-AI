from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    fbeta_score,
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

VALIDATION_REFERENCE = (
    DATA_DIR
    / "url_validation_v1.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "saved_models"
    / "url_binary_v3_candidate.joblib"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "url_detection"
    / "binary_threshold_comparison_v3.csv"
)


def reconstruct_validation(
    features_df,
):
    reference = pd.read_csv(
        VALIDATION_REFERENCE,
        usecols=["url", "type"],
    )

    validation_urls = set(
        reference["url"]
    )

    validation = features_df[
        features_df["url"].isin(
            validation_urls
        )
    ].copy()

    if len(validation) != len(reference):
        raise ValueError(
            "Validation reconstruction mismatch: "
            f"expected {len(reference):,}, "
            f"found {len(validation):,}"
        )

    return validation


def evaluate_threshold(
    actual_threat,
    scores,
    threshold,
):
    predicted_threat = (
        scores >= threshold
    )

    precision = precision_score(
        actual_threat,
        predicted_threat,
        zero_division=0,
    )

    recall = recall_score(
        actual_threat,
        predicted_threat,
        zero_division=0,
    )

    f1 = f1_score(
        actual_threat,
        predicted_threat,
        zero_division=0,
    )

    f2 = fbeta_score(
        actual_threat,
        predicted_threat,
        beta=2,
        zero_division=0,
    )

    true_positive = int(
        (
            actual_threat
            & predicted_threat
        ).sum()
    )

    false_negative = int(
        (
            actual_threat
            & (~predicted_threat)
        ).sum()
    )

    false_positive = int(
        (
            (~actual_threat)
            & predicted_threat
        ).sum()
    )

    true_negative = int(
        (
            (~actual_threat)
            & (~predicted_threat)
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

    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "false_alarm_rate":
            false_alarm_rate,
        "threats_detected":
            true_positive,
        "threats_missed":
            false_negative,
        "benign_false_alarms":
            false_positive,
        "benign_correct":
            true_negative,
    }


def choose_for_limit(
    results_df,
    max_false_alarm,
):
    acceptable = results_df[
        results_df[
            "false_alarm_rate"
        ] <= max_false_alarm
    ].copy()

    if acceptable.empty:
        return None

    return (
        acceptable
        .sort_values(
            [
                "recall",
                "f2",
                "precision",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .iloc[0]
    )


def print_option(
    title,
    row,
):
    print("\n" + "-" * 72)
    print(title)
    print("-" * 72)

    if row is None:
        print(
            "No threshold satisfied "
            "this false-alarm constraint."
        )
        return

    print(
        f"Threshold             : "
        f"{row['threshold']:.2f}"
    )

    print(
        f"Threat Precision      : "
        f"{row['precision']:.4f}"
    )

    print(
        f"Threat Recall         : "
        f"{row['recall']:.4f}"
    )

    print(
        f"Threat F1             : "
        f"{row['f1']:.4f}"
    )

    print(
        f"Threat F2             : "
        f"{row['f2']:.4f}"
    )

    print(
        f"False Alarm Rate      : "
        f"{row['false_alarm_rate']:.4f}"
    )

    print(
        f"Threats detected      : "
        f"{int(row['threats_detected']):,}"
    )

    print(
        f"Threats missed        : "
        f"{int(row['threats_missed']):,}"
    )

    print(
        f"Benign false alarms   : "
        f"{int(row['benign_false_alarms']):,}"
    )


def main():
    print("=" * 78)
    print(
        "DeepShield AI - "
        "Binary Threat Threshold Tuning v3"
    )
    print("=" * 78)

    package = joblib.load(
        MODEL_PATH
    )

    model = package["model"]

    feature_columns = package[
        "feature_columns"
    ]

    print(
        f"\nModel: "
        f"{package['model_name']}"
    )

    print(
        "\nLoading feature dataset..."
    )

    features_df = pd.read_csv(
        FEATURES_PATH
    )

    validation = (
        reconstruct_validation(
            features_df
        )
    )

    print(
        f"Validation samples: "
        f"{len(validation):,}"
    )

    X = validation[
        feature_columns
    ]

    actual_threat = (
        validation["type"]
        != "benign"
    ).to_numpy()

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

    results = []

    for threshold in np.arange(
        0.10,
        0.91,
        0.01,
    ):
        results.append(
            evaluate_threshold(
                actual_threat,
                threat_scores,
                round(
                    float(threshold),
                    2,
                ),
            )
        )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    option_5 = choose_for_limit(
        results_df,
        0.05,
    )

    option_75 = choose_for_limit(
        results_df,
        0.075,
    )

    option_10 = choose_for_limit(
        results_df,
        0.10,
    )

    best_f2 = (
        results_df
        .sort_values(
            [
                "f2",
                "recall",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    print("\n" + "=" * 78)
    print("THRESHOLD OPTIONS")
    print("=" * 78)

    print_option(
        "Option A — Max 5% false alarms",
        option_5,
    )

    print_option(
        "Option B — Max 7.5% false alarms",
        option_75,
    )

    print_option(
        "Option C — Max 10% false alarms",
        option_10,
    )

    print_option(
        "Option D — Best F2 score",
        best_f2,
    )

    print("\n" + "=" * 78)
    print("REFERENCE — DEFAULT THRESHOLD 0.50")
    print("=" * 78)

    default_row = results_df[
        results_df[
            "threshold"
        ] == 0.50
    ].iloc[0]

    print_option(
        "Default 0.50",
        default_row,
    )

    print(
        f"\nSaved:\n{OUTPUT_PATH}"
    )

    print(
        "\nThresholds were evaluated "
        "using validation data only."
    )


if __name__ == "__main__":
    main()