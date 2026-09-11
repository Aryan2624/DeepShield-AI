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


TEST_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "phishing_urls"
    / "url_test.csv"
)


MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "saved_models"
    / "url_detector_candidate.joblib"
)


RESULT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "url_detection"
    / "evaluation"
)


METRICS_PATH = (
    RESULT_DIR
    / "test_metrics.json"
)


CLASS_REPORT_PATH = (
    RESULT_DIR
    / "classification_report.csv"
)


CONFUSION_MATRIX_PATH = (
    RESULT_DIR
    / "confusion_matrix.csv"
)


def evaluate_model():
    print("=" * 75)
    print("DeepShield AI - Final URL Model Test Evaluation")
    print("=" * 75)


    # -------------------------------------------------
    # Validate files
    # -------------------------------------------------

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Candidate model not found:\n{MODEL_PATH}"
        )


    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -------------------------------------------------
    # Load model package
    # -------------------------------------------------

    print("\nLoading candidate model...")

    model_package = joblib.load(
        MODEL_PATH
    )


    model = model_package["model"]

    model_name = model_package[
        "model_name"
    ]

    feature_columns = model_package[
        "feature_columns"
    ]


    print(
        f"Selected model: "
        f"{model_name}"
    )


    # -------------------------------------------------
    # Load untouched test set
    # -------------------------------------------------

    print("\nLoading untouched test dataset...")

    test_df = pd.read_csv(
        TEST_PATH
    )


    missing_features = [
        feature
        for feature in feature_columns
        if feature not in test_df.columns
    ]


    if missing_features:
        raise ValueError(
            "Test dataset is missing features: "
            + ", ".join(missing_features)
        )


    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        "type"
    ]


    print(
        f"Test samples : "
        f"{len(X_test):,}"
    )

    print(
        f"Features     : "
        f"{len(feature_columns)}"
    )


    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

    print("\nGenerating final test predictions...")

    predictions = model.predict(
        X_test
    )


    # -------------------------------------------------
    # Overall metrics
    # -------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )


    precision_macro = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )


    recall_macro = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )


    f1_macro = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )


    precision_weighted = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )


    recall_weighted = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )


    f1_weighted = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )


    # -------------------------------------------------
    # ROC-AUC
    # -------------------------------------------------

    roc_auc_macro = None

    if hasattr(
        model,
        "predict_proba",
    ):

        probabilities = model.predict_proba(
            X_test
        )

        try:

            roc_auc_macro = roc_auc_score(
                y_test,
                probabilities,
                labels=model.classes_,
                multi_class="ovr",
                average="macro",
            )

        except ValueError:

            roc_auc_macro = None


    # -------------------------------------------------
    # Classification report
    # -------------------------------------------------

    report_dict = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )


    report_df = (
        pd.DataFrame(report_dict)
        .transpose()
    )


    # -------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------

    class_labels = list(
        model.classes_
    )


    cm = confusion_matrix(
        y_test,
        predictions,
        labels=class_labels,
    )


    confusion_df = pd.DataFrame(
        cm,
        index=[
            f"Actual_{label}"
            for label in class_labels
        ],
        columns=[
            f"Predicted_{label}"
            for label in class_labels
        ],
    )


    # -------------------------------------------------
    # Save results
    # -------------------------------------------------

    metrics = {
        "model_name":
            model_name,

        "test_samples":
            int(len(X_test)),

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
            (
                float(roc_auc_macro)
                if roc_auc_macro is not None
                else None
            ),

        "classes":
            class_labels,
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
        CLASS_REPORT_PATH
    )


    confusion_df.to_csv(
        CONFUSION_MATRIX_PATH
    )


    # -------------------------------------------------
    # Display results
    # -------------------------------------------------

    print("\n" + "=" * 75)
    print("FINAL TEST METRICS")
    print("=" * 75)


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
        f"Precision Weighted  : "
        f"{precision_weighted:.4f}"
    )

    print(
        f"Recall Weighted     : "
        f"{recall_weighted:.4f}"
    )

    print(
        f"F1 Weighted         : "
        f"{f1_weighted:.4f}"
    )


    if roc_auc_macro is not None:

        print(
            f"ROC-AUC Macro       : "
            f"{roc_auc_macro:.4f}"
        )


    print("\n" + "=" * 75)
    print("PER-CLASS PERFORMANCE")
    print("=" * 75)


    display_report = report_df.loc[
        class_labels,
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


    print("\n" + "=" * 75)
    print("CONFUSION MATRIX")
    print("=" * 75)


    print(
        confusion_df.to_string()
    )


    print("\n" + "=" * 75)
    print("EVALUATION FILES")
    print("=" * 75)


    print(
        f"\nMetrics:\n"
        f"{METRICS_PATH}"
    )

    print(
        f"\nClassification report:\n"
        f"{CLASS_REPORT_PATH}"
    )

    print(
        f"\nConfusion matrix:\n"
        f"{CONFUSION_MATRIX_PATH}"
    )


    print(
        "\nThe final test set has now been used "
        "for final evaluation."
    )


if __name__ == "__main__":
    evaluate_model()