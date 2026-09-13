import time
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import extract_url_features


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "phishing_urls"
)

FEATURES_PATH = DATA_DIR / "url_features_v2.csv"

TRAIN_REFERENCE = DATA_DIR / "url_train_v1.csv"

VALIDATION_REFERENCE = (
    DATA_DIR
    / "url_validation_v1.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "saved_models"
)

CANDIDATE_PATH = (
    MODEL_DIR
    / "url_binary_v3_candidate.joblib"
)

COMPARISON_PATH = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "url_detection"
    / "binary_model_comparison_v3.csv"
)


BENIGN_SANITY_URLS = [
    "https://www.google.com",
    "https://github.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.wikipedia.org",
    "https://www.python.org",
]


SUSPICIOUS_SANITY_URLS = [
    "http://192.168.1.20/login/verify/account",
    "http://secure-login-verify-account.example.com/password/reset",
]


def reconstruct_split(
    features_df,
    reference_path,
    split_name,
):
    print(f"Reconstructing {split_name}...")

    reference = pd.read_csv(
        reference_path,
        usecols=["url", "type"],
    )

    urls = set(reference["url"])

    split_df = features_df[
        features_df["url"].isin(urls)
    ].copy()

    if len(split_df) != len(reference):
        raise ValueError(
            f"{split_name} mismatch: "
            f"expected {len(reference):,}, "
            f"found {len(split_df):,}"
        )

    print(
        f"{split_name} samples : "
        f"{len(split_df):,}"
    )

    return split_df


def make_binary_labels(labels):
    return (
        labels != "benign"
    ).astype(int)


def prepare_url(
    url,
    feature_columns,
):
    features = extract_url_features(url)

    return pd.DataFrame(
        [[
            features[column]
            for column in feature_columns
        ]],
        columns=feature_columns,
    )


def sanity_check(
    model,
    feature_columns,
):
    print("\nDevelopment sanity check")
    print("-" * 80)

    benign_correct = 0

    for url in BENIGN_SANITY_URLS:

        X = prepare_url(
            url,
            feature_columns,
        )

        probability = float(
            model.predict_proba(X)[0][1]
        )

        prediction = int(
            probability >= 0.50
        )

        if prediction == 0:
            benign_correct += 1

        label = (
            "THREAT"
            if prediction == 1
            else "BENIGN"
        )

        print(
            f"{url:<40} "
            f"→ {label:<8} "
            f"Threat score: "
            f"{probability * 100:6.2f}%"
        )

    suspicious_correct = 0

    print("\nSynthetic suspicious URLs:")

    for url in SUSPICIOUS_SANITY_URLS:

        X = prepare_url(
            url,
            feature_columns,
        )

        probability = float(
            model.predict_proba(X)[0][1]
        )

        prediction = int(
            probability >= 0.50
        )

        if prediction == 1:
            suspicious_correct += 1

        label = (
            "THREAT"
            if prediction == 1
            else "BENIGN"
        )

        print(
            f"{url:<65} "
            f"→ {label:<8} "
            f"Threat score: "
            f"{probability * 100:6.2f}%"
        )

    print(
        "\nKnown benign roots classified benign : "
        f"{benign_correct}/"
        f"{len(BENIGN_SANITY_URLS)}"
    )

    print(
        "Synthetic suspicious URLs flagged    : "
        f"{suspicious_correct}/"
        f"{len(SUSPICIOUS_SANITY_URLS)}"
    )

    return (
        benign_correct,
        suspicious_correct,
    )


def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    feature_columns,
):
    print("\n" + "=" * 80)
    print(f"TRAINING: {name}")
    print("=" * 80)

    start = time.time()

    model.fit(
        X_train,
        y_train,
    )

    training_time = (
        time.time() - start
    )

    probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    f2 = fbeta_score(
        y_validation,
        predictions,
        beta=2,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_validation,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_validation,
        probabilities,
    )

    false_positive = int(
        (
            (y_validation == 0)
            & (predictions == 1)
        ).sum()
    )

    true_negative = int(
        (
            (y_validation == 0)
            & (predictions == 0)
        ).sum()
    )

    false_negative = int(
        (
            (y_validation == 1)
            & (predictions == 0)
        ).sum()
    )

    true_positive = int(
        (
            (y_validation == 1)
            & (predictions == 1)
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

    print(
        f"\nTraining time          : "
        f"{training_time:.2f} sec"
    )

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

    print(
        f"Threats detected       : "
        f"{true_positive:,}"
    )

    print(
        f"Threats missed         : "
        f"{false_negative:,}"
    )

    (
        benign_sanity,
        suspicious_sanity,
    ) = sanity_check(
        model,
        feature_columns,
    )

    return {
        "model": name,
        "training_time_seconds":
            training_time,
        "accuracy":
            accuracy,
        "precision":
            precision,
        "recall":
            recall,
        "f1":
            f1,
        "f2":
            f2,
        "roc_auc":
            roc_auc,
        "pr_auc":
            pr_auc,
        "false_alarm_rate":
            false_alarm_rate,
        "threats_detected":
            true_positive,
        "threats_missed":
            false_negative,
        "benign_sanity_safe":
            benign_sanity,
        "suspicious_sanity_flagged":
            suspicious_sanity,
    }


def main():
    print("=" * 80)
    print(
        "DeepShield AI - "
        "URL Binary Threat Detector v3"
    )
    print("=" * 80)

    required_files = [
        FEATURES_PATH,
        TRAIN_REFERENCE,
        VALIDATION_REFERENCE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{path}"
            )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nLoading v2 feature dataset..."
    )

    features_df = pd.read_csv(
        FEATURES_PATH
    )

    print(
        f"Total rows : "
        f"{len(features_df):,}"
    )

    train_df = reconstruct_split(
        features_df,
        TRAIN_REFERENCE,
        "TRAIN",
    )

    validation_df = reconstruct_split(
        features_df,
        VALIDATION_REFERENCE,
        "VALIDATION",
    )

    overlap = len(
        set(train_df["url"])
        &
        set(validation_df["url"])
    )

    print(
        f"\nTrain ↔ Validation overlap : "
        f"{overlap}"
    )

    if overlap != 0:
        raise ValueError(
            "Data leakage detected."
        )

    feature_columns = [
        column
        for column
        in features_df.columns
        if column not in {
            "url",
            "type",
        }
    ]

    X_train = train_df[
        feature_columns
    ]

    X_validation = validation_df[
        feature_columns
    ]

    y_train = make_binary_labels(
        train_df["type"]
    )

    y_validation = make_binary_labels(
        validation_df["type"]
    )

    print(
        f"\nTraining samples   : "
        f"{len(X_train):,}"
    )

    print(
        f"Validation samples : "
        f"{len(X_validation):,}"
    )

    print(
        f"Features           : "
        f"{len(feature_columns)}"
    )

    print(
        "\nTraining label distribution:"
    )

    print(
        y_train
        .value_counts()
        .rename(
            index={
                0: "benign",
                1: "threat",
            }
        )
        .to_string()
    )

    models = [
        (
            "Logistic Regression Standard",
            Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),

        (
            "Random Forest Standard",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                min_samples_split=10,
                min_samples_leaf=2,
                class_weight=None,
                n_jobs=-1,
                random_state=42,
            ),
        ),

        (
            "Random Forest Balanced",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                min_samples_split=10,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
        ),

        (
            "Hist Gradient Boosting Standard",
            HistGradientBoostingClassifier(
                learning_rate=0.1,
                max_iter=200,
                max_leaf_nodes=31,
                class_weight=None,
                random_state=42,
            ),
        ),

        (
            "Hist Gradient Boosting Balanced",
            HistGradientBoostingClassifier(
                learning_rate=0.1,
                max_iter=200,
                max_leaf_nodes=31,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]

    results = []

    best_pr_auc = -1.0
    best_model_name = None

    for (
        model_name,
        model,
    ) in models:

        metrics = evaluate_model(
            model_name,
            model,
            X_train,
            y_train,
            X_validation,
            y_validation,
            feature_columns,
        )

        results.append(metrics)

        if (
            metrics["pr_auc"]
            > best_pr_auc
        ):
            best_pr_auc = (
                metrics["pr_auc"]
            )

            best_model_name = (
                model_name
            )

            package = {
                "model":
                    model,
                "model_name":
                    model_name,
                "version":
                    "3.0-binary-candidate",
                "task":
                    "binary_url_threat_detection",
                "feature_engine":
                    "v2",
                "feature_columns":
                    feature_columns,
                "feature_count":
                    len(feature_columns),
                "label_mapping": {
                    "benign": 0,
                    "threat": 1,
                },
                "selection_metric":
                    "pr_auc",
                "validation_pr_auc":
                    best_pr_auc,
                "random_state":
                    42,
            }

            joblib.dump(
                package,
                CANDIDATE_PATH,
                compress=3,
            )

            print(
                "\nCurrent best binary "
                f"candidate saved: "
                f"{model_name}"
            )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "pr_auc",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    results_df.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    print("\n" + "=" * 80)
    print("V3 BINARY MODEL COMPARISON")
    print("=" * 80)

    display_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f2",
        "roc_auc",
        "pr_auc",
        "false_alarm_rate",
        "benign_sanity_safe",
        "suspicious_sanity_flagged",
    ]

    print(
        results_df[
            display_columns
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("V3 BINARY CANDIDATE")
    print("=" * 80)

    print(
        f"Best by PR-AUC : "
        f"{best_model_name}"
    )

    print(
        f"PR-AUC         : "
        f"{best_pr_auc:.4f}"
    )

    print(
        "\nThis is a development candidate."
    )

    print(
        "Its threat threshold still needs "
        "to be tuned using validation data."
    )

    print(
        f"\nCandidate model:\n"
        f"{CANDIDATE_PATH}"
    )

    print(
        f"\nComparison CSV:\n"
        f"{COMPARISON_PATH}"
    )


if __name__ == "__main__":
    main()