from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42


PROJECT_ROOT = Path(__file__).resolve().parents[3]


TRAIN_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "phishing_urls"
    / "url_train.csv"
)


VALIDATION_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "phishing_urls"
    / "url_validation.csv"
)


MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "saved_models"
)


RESULTS_PATH = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "url_detection"
    / "model_comparison.csv"
)


BEST_MODEL_PATH = (
    MODEL_DIR
    / "url_detector_candidate.joblib"
)


def load_data():

    print("Loading training data...")

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    print("Loading validation data...")

    validation_df = pd.read_csv(
        VALIDATION_PATH
    )


    feature_columns = [
        column
        for column in train_df.columns
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


    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns,
    )


def build_models():

    models = {

        "Logistic Regression":
            Pipeline(
                steps=[
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),


        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=25,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),


        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                min_samples_split=10,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),


        "Hist Gradient Boosting":
            HistGradientBoostingClassifier(
                learning_rate=0.1,
                max_iter=200,
                max_leaf_nodes=31,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
    }

    return models


def calculate_metrics(
    model,
    X_validation,
    y_validation,
):

    predictions = model.predict(
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


    precision_weighted = precision_score(
        y_validation,
        predictions,
        average="weighted",
        zero_division=0,
    )


    recall_weighted = recall_score(
        y_validation,
        predictions,
        average="weighted",
        zero_division=0,
    )


    f1_weighted = f1_score(
        y_validation,
        predictions,
        average="weighted",
        zero_division=0,
    )


    roc_auc_macro = None


    if hasattr(
        model,
        "predict_proba",
    ):

        probabilities = model.predict_proba(
            X_validation
        )

        try:

            roc_auc_macro = roc_auc_score(
                y_validation,
                probabilities,
                labels=model.classes_,
                multi_class="ovr",
                average="macro",
            )

        except ValueError:

            roc_auc_macro = None


    return {
        "accuracy":
            accuracy,

        "precision_macro":
            precision_macro,

        "recall_macro":
            recall_macro,

        "f1_macro":
            f1_macro,

        "precision_weighted":
            precision_weighted,

        "recall_weighted":
            recall_weighted,

        "f1_weighted":
            f1_weighted,

        "roc_auc_macro":
            roc_auc_macro,
    }


def train_models():

    print("=" * 75)
    print("DeepShield AI - Malicious URL Model Training")
    print("=" * 75)


    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{TRAIN_PATH}"
        )


    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation dataset not found:\n{VALIDATION_PATH}"
        )


    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns,
    ) = load_data()


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


    models = build_models()


    results = []


    best_model = None

    best_model_name = None

    best_f1_macro = -1.0


    for model_name, model in models.items():

        print("\n" + "=" * 75)

        print(
            f"Training: "
            f"{model_name}"
        )

        print("=" * 75)


        start_time = perf_counter()


        model.fit(
            X_train,
            y_train,
        )


        training_time = (
            perf_counter()
            - start_time
        )


        print(
            f"Training completed in "
            f"{training_time:.2f} seconds"
        )


        metrics = calculate_metrics(
            model,
            X_validation,
            y_validation,
        )


        result = {
            "model":
                model_name,

            "training_time_seconds":
                round(
                    training_time,
                    2,
                ),

            **metrics,
        }


        results.append(
            result
        )


        print(
            f"Accuracy          : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision Macro   : "
            f"{metrics['precision_macro']:.4f}"
        )

        print(
            f"Recall Macro      : "
            f"{metrics['recall_macro']:.4f}"
        )

        print(
            f"F1 Macro          : "
            f"{metrics['f1_macro']:.4f}"
        )

        print(
            f"F1 Weighted       : "
            f"{metrics['f1_weighted']:.4f}"
        )


        if metrics["roc_auc_macro"] is not None:

            print(
                f"ROC-AUC Macro     : "
                f"{metrics['roc_auc_macro']:.4f}"
            )


        # -------------------------------------
        # Select best model using Macro F1
        # -------------------------------------

        if (
            metrics["f1_macro"]
            > best_f1_macro
        ):

            best_f1_macro = (
                metrics["f1_macro"]
            )

            best_model = model

            best_model_name = (
                model_name
            )


    # -----------------------------------------
    # Save comparison results
    # -----------------------------------------

    results_df = pd.DataFrame(
        results
    )


    results_df = (
        results_df
        .sort_values(
            by="f1_macro",
            ascending=False,
        )
        .reset_index(drop=True)
    )


    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )


    # -----------------------------------------
    # Save best validation model candidate
    # -----------------------------------------

    model_package = {

        "model":
            best_model,

        "model_name":
            best_model_name,

        "feature_columns":
            feature_columns,

        "selection_metric":
            "macro_f1",

        "validation_macro_f1":
            best_f1_macro,

        "random_state":
            RANDOM_STATE,
    }


    joblib.dump(
        model_package,
        BEST_MODEL_PATH,
    )


    print("\n" + "=" * 75)
    print("MODEL COMPARISON")
    print("=" * 75)


    display_columns = [
        "model",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "roc_auc_macro",
        "training_time_seconds",
    ]


    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )


    print("\n" + "=" * 75)
    print("BEST VALIDATION MODEL")
    print("=" * 75)


    print(
        f"Model    : "
        f"{best_model_name}"
    )

    print(
        f"Macro F1 : "
        f"{best_f1_macro:.4f}"
    )


    print(
        f"\nCandidate model saved to:\n"
        f"{BEST_MODEL_PATH}"
    )


    print(
        f"\nComparison results saved to:\n"
        f"{RESULTS_PATH}"
    )


    print(
        "\nIMPORTANT:"
        "\nThe final test set has NOT been used."
    )


if __name__ == "__main__":
    train_models()