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

TRAIN_REFERENCE = (
    DATA_DIR
    / "url_train_v1.csv"
)

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
    / "url_threat_type_v3_candidate.joblib"
)

COMPARISON_PATH = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "url_detection"
    / "stage2_model_comparison_v3.csv"
)


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

    reference_urls = set(
        reference["url"]
    )

    split_df = features_df[
        features_df["url"].isin(
            reference_urls
        )
    ].copy()

    if len(split_df) != len(reference):
        raise ValueError(
            f"{split_name} mismatch: "
            f"expected {len(reference):,}, "
            f"found {len(split_df):,}"
        )

    # Stage 2 trains ONLY on threat classes
    split_df = split_df[
        split_df["type"] != "benign"
    ].copy()

    print(
        f"{split_name} threat samples : "
        f"{len(split_df):,}"
    )

    print(
        split_df["type"]
        .value_counts()
        .to_string()
    )

    return split_df


def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
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

    probabilities = model.predict_proba(
        X_validation
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

    phishing_f1 = (
        report["phishing"]["f1-score"]
    )

    malware_f1 = (
        report["malware"]["f1-score"]
    )

    defacement_f1 = (
        report["defacement"]["f1-score"]
    )

    phishing_precision = (
        report["phishing"]["precision"]
    )

    phishing_recall = (
        report["phishing"]["recall"]
    )

    print(
        f"\nTraining time       : "
        f"{training_time:.2f} sec"
    )

    print(
        f"Accuracy            : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision Macro     : "
        f"{precision_macro:.4f}"
    )

    print(
        f"Recall Macro        : "
        f"{recall_macro:.4f}"
    )

    print(
        f"F1 Macro            : "
        f"{f1_macro:.4f}"
    )

    print(
        f"F1 Weighted         : "
        f"{f1_weighted:.4f}"
    )

    print(
        f"ROC-AUC Macro       : "
        f"{roc_auc_macro:.4f}"
    )

    print(
        f"Phishing Precision  : "
        f"{phishing_precision:.4f}"
    )

    print(
        f"Phishing Recall     : "
        f"{phishing_recall:.4f}"
    )

    print(
        f"Phishing F1         : "
        f"{phishing_f1:.4f}"
    )

    print(
        f"Malware F1          : "
        f"{malware_f1:.4f}"
    )

    print(
        f"Defacement F1       : "
        f"{defacement_f1:.4f}"
    )

    print("\nPer-class validation performance:")

    display_report = pd.DataFrame(
        report
    ).transpose().loc[
        list(model.classes_),
        [
            "precision",
            "recall",
            "f1-score",
            "support",
        ],
    ]

    print(
        display_report
        .round(4)
        .to_string()
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
        "phishing_precision":
            phishing_precision,
        "phishing_recall":
            phishing_recall,
        "phishing_f1":
            phishing_f1,
        "malware_f1":
            malware_f1,
        "defacement_f1":
            defacement_f1,
    }


def main():
    print("=" * 80)
    print(
        "DeepShield AI - "
        "URL Threat Type Classifier v3 Stage 2"
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
        f"Total feature rows : "
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
        for column in features_df.columns
        if column not in {
            "url",
            "type",
        }
    ]

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        "type"
    ]

    X_validation = validation_df[
        feature_columns
    ]

    y_validation = validation_df[
        "type"
    ]

    print(
        f"\nTraining threat samples   : "
        f"{len(X_train):,}"
    )

    print(
        f"Validation threat samples : "
        f"{len(X_validation):,}"
    )

    print(
        f"Features                  : "
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

    best_f1 = -1.0
    best_model_name = None

    for model_name, model in models:

        metrics = evaluate_model(
            model_name,
            model,
            X_train,
            y_train,
            X_validation,
            y_validation,
        )

        results.append(
            metrics
        )

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
                "model":
                    model,
                "model_name":
                    model_name,
                "version":
                    "3.0-stage2-candidate",
                "task":
                    "url_threat_type_classification",
                "feature_engine":
                    "v2",
                "feature_columns":
                    feature_columns,
                "feature_count":
                    len(feature_columns),
                "classes":
                    list(
                        model.classes_
                    ),
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
                "\nCurrent best Stage 2 "
                f"candidate saved: "
                f"{model_name}"
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
    print("V3 STAGE 2 MODEL COMPARISON")
    print("=" * 80)

    display_columns = [
        "model",
        "accuracy",
        "f1_macro",
        "roc_auc_macro",
        "phishing_precision",
        "phishing_recall",
        "phishing_f1",
        "malware_f1",
        "defacement_f1",
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
    print("V3 STAGE 2 CANDIDATE")
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
        "\nBenchmark split was NOT "
        "used during Stage 2 selection."
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