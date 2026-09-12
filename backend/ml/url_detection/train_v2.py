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
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import extract_url_features


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "phishing_urls"
)

FEATURES_V2_PATH = DATA_DIR / "url_features_v2.csv"

TRAIN_REFERENCE_PATH = DATA_DIR / "url_train_v1.csv"

VALIDATION_REFERENCE_PATH = (
    DATA_DIR / "url_validation_v1.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "saved_models"
)

CANDIDATE_PATH = (
    MODEL_DIR
    / "url_detector_v2_candidate.joblib"
)

COMPARISON_PATH = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "url_detection"
    / "model_comparison_v2.csv"
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


def build_split_from_reference(
    feature_df,
    reference_path,
    split_name,
):
    print(
        f"Reconstructing {split_name}..."
    )

    reference_df = pd.read_csv(
        reference_path,
        usecols=["url", "type"],
    )

    reference_urls = set(
        reference_df["url"]
    )

    split_df = feature_df[
        feature_df["url"].isin(
            reference_urls
        )
    ].copy()

    if len(split_df) != len(reference_df):
        raise ValueError(
            f"{split_name} mismatch: "
            f"expected {len(reference_df):,}, "
            f"found {len(split_df):,}"
        )

    reference_labels = (
        reference_df
        .set_index("url")["type"]
        .to_dict()
    )

    incorrect_labels = split_df[
        split_df.apply(
            lambda row:
                reference_labels.get(
                    row["url"]
                )
                != row["type"],
            axis=1,
        )
    ]

    if not incorrect_labels.empty:
        raise ValueError(
            f"{split_name} contains "
            "label mismatches."
        )

    print(
        f"{split_name} samples : "
        f"{len(split_df):,}"
    )

    return split_df


def prepare_url(
    url,
    feature_columns,
):
    features = extract_url_features(
        url
    )

    return pd.DataFrame(
        [
            [
                features[column]
                for column
                in feature_columns
            ]
        ],
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

        prediction = str(
            model.predict(X)[0]
        )

        probabilities = (
            model.predict_proba(X)[0]
        )

        confidence = float(
            max(probabilities)
        )

        if prediction == "benign":
            benign_correct += 1

        print(
            f"{url:<40} "
            f"→ {prediction.upper():<12} "
            f"{confidence * 100:6.2f}%"
        )

    suspicious_correct = 0

    print("\nSynthetic suspicious URLs:")

    for url in SUSPICIOUS_SANITY_URLS:
        X = prepare_url(
            url,
            feature_columns,
        )

        prediction = str(
            model.predict(X)[0]
        )

        probabilities = (
            model.predict_proba(X)[0]
        )

        confidence = float(
            max(probabilities)
        )

        if prediction != "benign":
            suspicious_correct += 1

        print(
            f"{url:<65} "
            f"→ {prediction.upper():<12} "
            f"{confidence * 100:6.2f}%"
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
        time.time()
        - start
    )

    predictions = model.predict(
        X_validation
    )

    probabilities = (
        model.predict_proba(
            X_validation
        )
    )

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    precision_macro = precision_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_validation,
        predictions,
        average="weighted",
        zero_division=0,
    )

    roc_auc_macro = roc_auc_score(
        y_validation,
        probabilities,
        labels=model.classes_,
        multi_class="ovr",
        average="macro",
    )

    report = classification_report(
        y_validation,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    benign_recall = (
        report["benign"]["recall"]
    )

    benign_false_positive_rate = (
        1.0 - benign_recall
    )

    phishing_precision = (
        report["phishing"]["precision"]
    )

    phishing_recall = (
        report["phishing"]["recall"]
    )

    phishing_f1 = (
        report["phishing"]["f1-score"]
    )

    malware_f1 = (
        report["malware"]["f1-score"]
    )

    print(
        f"\nTraining time         : "
        f"{training_time:.2f} sec"
    )

    print(
        f"Accuracy              : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision Macro       : "
        f"{precision_macro:.4f}"
    )

    print(
        f"Recall Macro          : "
        f"{recall_macro:.4f}"
    )

    print(
        f"F1 Macro              : "
        f"{f1_macro:.4f}"
    )

    print(
        f"F1 Weighted           : "
        f"{f1_weighted:.4f}"
    )

    print(
        f"ROC-AUC Macro         : "
        f"{roc_auc_macro:.4f}"
    )

    print(
        f"Benign Recall         : "
        f"{benign_recall:.4f}"
    )

    print(
        f"Benign False Positive : "
        f"{benign_false_positive_rate:.4f}"
    )

    print(
        f"Phishing Precision    : "
        f"{phishing_precision:.4f}"
    )

    print(
        f"Phishing Recall       : "
        f"{phishing_recall:.4f}"
    )

    print(
        f"Phishing F1           : "
        f"{phishing_f1:.4f}"
    )

    print(
        f"Malware F1            : "
        f"{malware_f1:.4f}"
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
        "precision_macro":
            precision_macro,
        "recall_macro":
            recall_macro,
        "f1_macro":
            f1_macro,
        "f1_weighted":
            f1_weighted,
        "roc_auc_macro":
            roc_auc_macro,
        "benign_recall":
            benign_recall,
        "benign_false_positive_rate":
            benign_false_positive_rate,
        "phishing_precision":
            phishing_precision,
        "phishing_recall":
            phishing_recall,
        "phishing_f1":
            phishing_f1,
        "malware_f1":
            malware_f1,
        "benign_sanity_safe":
            benign_sanity,
        "suspicious_sanity_flagged":
            suspicious_sanity,
    }


def main():
    print("=" * 80)
    print(
        "DeepShield AI - "
        "URL Detector v2 Model Training"
    )
    print("=" * 80)

    required_files = [
        FEATURES_V2_PATH,
        TRAIN_REFERENCE_PATH,
        VALIDATION_REFERENCE_PATH,
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
        "\nLoading complete "
        "v2 feature dataset..."
    )

    features_v2 = pd.read_csv(
        FEATURES_V2_PATH
    )

    print(
        f"Total feature rows : "
        f"{len(features_v2):,}"
    )

    print()

    train_df = (
        build_split_from_reference(
            features_v2,
            TRAIN_REFERENCE_PATH,
            "TRAIN",
        )
    )

    validation_df = (
        build_split_from_reference(
            features_v2,
            VALIDATION_REFERENCE_PATH,
            "VALIDATION",
        )
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
        for column in features_v2.columns
        if column not in {
            "url",
            "type",
        }
    ]

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df["type"]

    X_validation = validation_df[
        feature_columns
    ]

    y_validation = validation_df[
        "type"
    ]

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

    models = [
        (
            "Logistic Regression Balanced",
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
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),

        (
            "Decision Tree Balanced",
            DecisionTreeClassifier(
                max_depth=25,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight="balanced",
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
            "Hist Gradient Boosting Balanced",
            HistGradientBoostingClassifier(
                learning_rate=0.1,
                max_iter=200,
                max_leaf_nodes=31,
                class_weight="balanced",
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
    ]

    results = []

    best_f1 = -1.0
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
            metrics["f1_macro"]
            > best_f1
        ):
            best_f1 = (
                metrics["f1_macro"]
            )

            best_model_name = (
                model_name
            )

            package = {
                "model": model,
                "model_name":
                    model_name,
                "version":
                    "2.0-candidate",
                "feature_engine":
                    "v2",
                "feature_columns":
                    feature_columns,
                "feature_count":
                    len(feature_columns),
                "selection_metric":
                    "macro_f1",
                "validation_macro_f1":
                    best_f1,
                "random_state":
                    42,
            }

            joblib.dump(
                package,
                CANDIDATE_PATH,
                compress=3,
            )

            print(
                "\nCurrent best candidate "
                f"saved: {model_name}"
            )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "f1_macro",
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
    print("V2 MODEL COMPARISON")
    print("=" * 80)

    columns = [
        "model",
        "accuracy",
        "f1_macro",
        "roc_auc_macro",
        "benign_recall",
        "phishing_precision",
        "phishing_f1",
        "malware_f1",
        "benign_sanity_safe",
        "suspicious_sanity_flagged",
    ]

    print(
        results_df[
            columns
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("VALIDATION CANDIDATE")
    print("=" * 80)

    print(
        f"Best by Macro F1 : "
        f"{best_model_name}"
    )

    print(
        f"Macro F1         : "
        f"{best_f1:.4f}"
    )

    print(
        "\nThe benchmark/test split "
        "was NOT used during training."
    )

    print(
        "Candidate status remains "
        "development-only until "
        "sanity and benchmark review."
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