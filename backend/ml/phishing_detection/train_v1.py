from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    r"E:\DeepShield-Datasets\phishing_messages\processed"
)

MODEL_DIR = Path(
    r"C:\Users\dubey\DeepShield-AI\backend\ml\saved_models"
)

OUTPUT_DIR = Path(
    r"C:\Users\dubey\DeepShield-AI\backend\ml\phishing_detection"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TRAIN_PATH = (
    DATASET_DIR
    / "phishing_train_v1.csv"
)

VALIDATION_PATH = (
    DATASET_DIR
    / "phishing_validation_v1.csv"
)


RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 90)
print("DEEPSHIELD AI - PHISHING DETECTOR V1")
print("=" * 90)

print("\nLoading datasets...")


train_df = pd.read_csv(
    TRAIN_PATH,
    low_memory=False,
)

validation_df = pd.read_csv(
    VALIDATION_PATH,
    low_memory=False,
)


X_train_text = (
    train_df["text"]
    .fillna("")
    .astype(str)
)

y_train = (
    train_df["label"]
    .astype(int)
)


X_validation_text = (
    validation_df["text"]
    .fillna("")
    .astype(str)
)

y_validation = (
    validation_df["label"]
    .astype(int)
)


print(
    f"Training samples:   "
    f"{len(train_df):,}"
)

print(
    f"Validation samples: "
    f"{len(validation_df):,}"
)


# ============================================================
# TF-IDF
# ============================================================

print("\n" + "=" * 90)
print("BUILDING TF-IDF FEATURES")
print("=" * 90)


vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",

    # Unigrams + bigrams
    ngram_range=(1, 2),

    # Ignore tokens appearing only once
    min_df=2,

    # Ignore extremely common terms
    max_df=0.995,

    # Prevent feature explosion
    max_features=120000,

    # Gives less weight to very frequent terms
    sublinear_tf=True,

    # Standard word tokenization
    analyzer="word",
)


start_time = time.time()


X_train = vectorizer.fit_transform(
    X_train_text
)

X_validation = vectorizer.transform(
    X_validation_text
)


tfidf_time = (
    time.time()
    - start_time
)


print(
    f"TF-IDF vocabulary size: "
    f"{len(vectorizer.vocabulary_):,}"
)

print(
    f"Training matrix shape: "
    f"{X_train.shape}"
)

print(
    f"Validation matrix shape: "
    f"{X_validation.shape}"
)

print(
    f"TF-IDF build time: "
    f"{tfidf_time:.2f} seconds"
)


# ============================================================
# MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="liblinear",
    ),

    "Multinomial Naive Bayes": MultinomialNB(
        alpha=0.5,
    ),

    "Linear SVM": LinearSVC(
        C=1.0,
        random_state=RANDOM_STATE,
    ),
}


results = []

trained_models = {}


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

for model_name, model in models.items():

    print("\n" + "=" * 90)
    print(f"MODEL: {model_name}")
    print("=" * 90)


    start_time = time.time()


    model.fit(
        X_train,
        y_train,
    )


    training_time = (
        time.time()
        - start_time
    )


    predictions = model.predict(
        X_validation
    )


    # --------------------------------------------------------
    # SCORES FOR ROC-AUC
    # --------------------------------------------------------

    if hasattr(
        model,
        "predict_proba",
    ):
        threat_scores = (
            model
            .predict_proba(
                X_validation
            )[:, 1]
        )

    elif hasattr(
        model,
        "decision_function",
    ):
        threat_scores = (
            model
            .decision_function(
                X_validation
            )
        )

    else:
        threat_scores = predictions


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_validation,
        predictions,
        average="weighted",
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_validation,
        threat_scores,
    )


    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1],
    ).ravel()


    false_alarm_rate = (
        fp
        / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )


    print(
        f"\nAccuracy:              "
        f"{accuracy:.4f}"
    )

    print(
        f"Threat Precision:      "
        f"{precision:.4f}"
    )

    print(
        f"Threat Recall:         "
        f"{recall:.4f}"
    )

    print(
        f"Threat F1:             "
        f"{f1:.4f}"
    )

    print(
        f"Macro F1:              "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1:           "
        f"{weighted_f1:.4f}"
    )

    print(
        f"ROC-AUC:               "
        f"{roc_auc:.4f}"
    )

    print(
        f"False Alarm Rate:      "
        f"{false_alarm_rate:.4f}"
    )

    print(
        f"Training Time:         "
        f"{training_time:.2f} sec"
    )


    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1],
        )
    )


    print("\nClassification Report:")

    print(
        classification_report(
            y_validation,
            predictions,
            target_names=[
                "LEGITIMATE",
                "THREAT",
            ],
            digits=4,
            zero_division=0,
        )
    )


    results.append(
        {
            "model": model_name,
            "accuracy": accuracy,
            "threat_precision": precision,
            "threat_recall": recall,
            "threat_f1": f1,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "roc_auc": roc_auc,
            "false_alarm_rate": false_alarm_rate,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "training_time_seconds": training_time,
        }
    )


    trained_models[
        model_name
    ] = model


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = (
    results_df
    .sort_values(
        by=[
            "macro_f1",
            "threat_recall",
        ],
        ascending=False,
    )
    .reset_index(drop=True)
)


print("\n" + "=" * 90)
print("MODEL COMPARISON")
print("=" * 90)


display_columns = [
    "model",
    "accuracy",
    "threat_precision",
    "threat_recall",
    "threat_f1",
    "macro_f1",
    "roc_auc",
    "false_alarm_rate",
]


print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = (
    results_df
    .iloc[0]["model"]
)


best_model = trained_models[
    best_model_name
]


print("\n" + "=" * 90)
print("SELECTED VALIDATION CANDIDATE")
print("=" * 90)

print(
    f"Best model by Macro F1: "
    f"{best_model_name}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

comparison_path = (
    OUTPUT_DIR
    / "model_comparison_v1.csv"
)


results_df.to_csv(
    comparison_path,
    index=False,
)


# ============================================================
# SAVE CANDIDATE PIPELINE
# ============================================================

candidate = {
    "vectorizer": vectorizer,
    "model": best_model,
    "model_name": best_model_name,
    "label_mapping": {
        0: "LEGITIMATE",
        1: "THREAT",
    },
    "version": "v1",
}


candidate_path = (
    MODEL_DIR
    / "phishing_detector_v1_candidate.joblib"
)


joblib.dump(
    candidate,
    candidate_path,
)


print(
    f"\nComparison saved to:\n"
    f"{comparison_path}"
)

print(
    f"\nCandidate model saved to:\n"
    f"{candidate_path}"
)


print("\n" + "=" * 90)
print("IMPORTANT")
print("=" * 90)

print(
    "The TEST dataset has NOT been used."
)

print(
    "Model selection is based only on the validation set."
)

print(
    "Do not report final performance until the selected pipeline "
    "is evaluated on the held-out test set."
)


print("\n" + "=" * 90)
print("TRAINING COMPLETE")
print("=" * 90)