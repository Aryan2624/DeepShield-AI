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

FEATURES_PATH = DATA_DIR / "url_features_v2.csv"

VALIDATION_REFERENCE = (
    DATA_DIR / "url_validation_v1.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "saved_models"
    / "url_detector_v2_candidate.joblib"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "url_detection"
    / "threshold_comparison_v2.csv"
)


def reconstruct_validation(features_df):
    reference = pd.read_csv(
        VALIDATION_REFERENCE,
        usecols=["url", "type"],
    )

    validation_urls = set(reference["url"])

    validation = features_df[
        features_df["url"].isin(validation_urls)
    ].copy()

    if len(validation) != len(reference):
        raise ValueError(
            f"Validation reconstruction mismatch: "
            f"expected {len(reference):,}, "
            f"found {len(validation):,}"
        )

    return validation


def main():
    print("=" * 78)
    print("DeepShield AI - Threat Threshold Tuning v2")
    print("=" * 78)

    package = joblib.load(MODEL_PATH)

    model = package["model"]
    feature_columns = package["feature_columns"]

    print(f"\nModel: {package['model_name']}")

    print("\nLoading feature dataset...")

    features_df = pd.read_csv(FEATURES_PATH)

    validation = reconstruct_validation(
        features_df
    )

    print(
        f"Validation samples: "
        f"{len(validation):,}"
    )

    X = validation[feature_columns]
    y = validation["type"]

    probabilities = model.predict_proba(X)

    benign_index = list(
        model.classes_
    ).index("benign")

    benign_probability = probabilities[
        :,
        benign_index
    ]

    threat_score = (
        1.0 - benign_probability
    )

    actual_threat = (
        y != "benign"
    ).to_numpy()

    results = []

    thresholds = np.arange(
        0.10,
        0.91,
        0.01,
    )

    for threshold in thresholds:

        predicted_threat = (
            threat_score >= threshold
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

        false_positive = (
            (~actual_threat)
            & predicted_threat
        ).sum()

        true_negative = (
            (~actual_threat)
            & (~predicted_threat)
        ).sum()

        false_negative = (
            actual_threat
            & (~predicted_threat)
        ).sum()

        true_positive = (
            actual_threat
            & predicted_threat
        ).sum()

        false_alarm_rate = (
            false_positive
            /
            (
                false_positive
                + true_negative
            )
        )

        results.append(
            {
                "threshold":
                    round(
                        float(threshold),
                        2,
                    ),
                "precision":
                    precision,
                "recall":
                    recall,
                "f1":
                    f1,
                "f2":
                    f2,
                "false_alarm_rate":
                    false_alarm_rate,
                "threats_detected":
                    int(true_positive),
                "threats_missed":
                    int(false_negative),
                "benign_false_alarms":
                    int(false_positive),
            }
        )

    results_df = pd.DataFrame(
        results
    )

    # Production-oriented constraint:
    # keep benign false alarms <= 5%
    acceptable = results_df[
        results_df[
            "false_alarm_rate"
        ] <= 0.05
    ].copy()

    if acceptable.empty:
        raise ValueError(
            "No threshold achieved "
            "false alarm rate <= 5%."
        )

    # Among acceptable thresholds,
    # prioritize threat recall.
    best = (
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

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 78)
    print("SELECTED THRESHOLD")
    print("=" * 78)

    print(
        f"Threat threshold       : "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Threat precision       : "
        f"{best['precision']:.4f}"
    )

    print(
        f"Threat recall          : "
        f"{best['recall']:.4f}"
    )

    print(
        f"Threat F1              : "
        f"{best['f1']:.4f}"
    )

    print(
        f"Threat F2              : "
        f"{best['f2']:.4f}"
    )

    print(
        f"Benign false alarm rate: "
        f"{best['false_alarm_rate']:.4f}"
    )

    print(
        f"Threats detected       : "
        f"{int(best['threats_detected']):,}"
    )

    print(
        f"Threats missed         : "
        f"{int(best['threats_missed']):,}"
    )

    print(
        f"Benign false alarms    : "
        f"{int(best['benign_false_alarms']):,}"
    )

    print("\n" + "=" * 78)
    print("TOP 10 VALIDATION THRESHOLDS")
    print("=" * 78)

    top = (
        acceptable
        .sort_values(
            [
                "recall",
                "f2",
            ],
            ascending=False,
        )
        .head(10)
    )

    print(
        top[
            [
                "threshold",
                "precision",
                "recall",
                "f1",
                "f2",
                "false_alarm_rate",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        f"\nSaved:\n{OUTPUT_PATH}"
    )

    print(
        "\nIMPORTANT: Threshold selected "
        "using validation data only."
    )


if __name__ == "__main__":
    main()