from pathlib import Path
import time

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path(
    r"E:\DeepShield-Datasets\phishing_messages\processed\phishing_master_v1.csv"
)

OUTPUT_PATH = Path(
    r"C:\Users\dubey\DeepShield-AI\backend\ml\phishing_detection\cross_source_char_v2.csv"
)

RANDOM_STATE = 42


MIXED_SOURCES = [
    "CEAS_08",
    "Enron",
    "Ling",
    "SpamAssasin",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 95)
print("DEEPSHIELD AI - CHARACTER TF-IDF CROSS-SOURCE TEST V2")
print("=" * 95)


df = pd.read_csv(
    DATASET_PATH,
    low_memory=False,
)


df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
)


print(
    f"\nTotal messages: "
    f"{len(df):,}"
)


results = []


# ============================================================
# LEAVE-ONE-SOURCE-OUT
# ============================================================

for held_out_source in MIXED_SOURCES:

    print("\n" + "=" * 95)

    print(
        f"HELD-OUT SOURCE: "
        f"{held_out_source}"
    )

    print("=" * 95)


    train_df = df[
        df["source"] != held_out_source
    ].copy()


    test_df = df[
        df["source"] == held_out_source
    ].copy()


    print(
        f"Training samples: "
        f"{len(train_df):,}"
    )

    print(
        f"Held-out samples: "
        f"{len(test_df):,}"
    )


    # ========================================================
    # CHARACTER TF-IDF
    # ========================================================

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",

        # Character sequences of length 3 to 5
        ngram_range=(3, 5),

        min_df=2,

        max_features=150000,

        sublinear_tf=True,

        lowercase=True,
    )


    start_time = time.time()


    X_train = vectorizer.fit_transform(
        train_df["text"]
    )


    X_test = vectorizer.transform(
        test_df["text"]
    )


    print(
        f"Character vocabulary: "
        f"{len(vectorizer.vocabulary_):,}"
    )


    # ========================================================
    # LINEAR SVM
    # ========================================================

    model = LinearSVC(
        C=1.0,
        random_state=RANDOM_STATE,
    )


    model.fit(
        X_train,
        train_df["label"].astype(int),
    )


    predictions = model.predict(
        X_test
    )


    elapsed = (
        time.time()
        - start_time
    )


    y_true = (
        test_df["label"]
        .astype(int)
    )


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        predictions,
    )


    precision = precision_score(
        y_true,
        predictions,
        pos_label=1,
        zero_division=0,
    )


    recall = recall_score(
        y_true,
        predictions,
        pos_label=1,
        zero_division=0,
    )


    threat_f1 = f1_score(
        y_true,
        predictions,
        pos_label=1,
        zero_division=0,
    )


    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )


    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()


    legitimate_recall = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )


    false_alarm_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )


    print(
        f"\nAccuracy:          "
        f"{accuracy:.4f}"
    )

    print(
        f"Threat Precision:  "
        f"{precision:.4f}"
    )

    print(
        f"Threat Recall:     "
        f"{recall:.4f}"
    )

    print(
        f"Threat F1:         "
        f"{threat_f1:.4f}"
    )

    print(
        f"Macro F1:          "
        f"{macro_f1:.4f}"
    )

    print(
        f"Legitimate Recall: "
        f"{legitimate_recall:.4f}"
    )

    print(
        f"False Alarm Rate:  "
        f"{false_alarm_rate:.4f}"
    )

    print(
        f"Elapsed Time:      "
        f"{elapsed:.2f} sec"
    )


    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        )
    )


    results.append(
        {
            "held_out_source": held_out_source,

            "accuracy": accuracy,

            "threat_precision": precision,

            "threat_recall": recall,

            "threat_f1": threat_f1,

            "macro_f1": macro_f1,

            "legitimate_recall": legitimate_recall,

            "false_alarm_rate": false_alarm_rate,

            "true_negative": int(tn),

            "false_positive": int(fp),

            "false_negative": int(fn),

            "true_positive": int(tp),

            "elapsed_seconds": elapsed,
        }
    )


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 95)
print("CHARACTER TF-IDF CROSS-SOURCE SUMMARY")
print("=" * 95)


summary_columns = [
    "held_out_source",
    "accuracy",
    "threat_precision",
    "threat_recall",
    "threat_f1",
    "macro_f1",
    "false_alarm_rate",
]


print(
    results_df[
        summary_columns
    ].to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)


print("\nAverage metrics:")


print(
    f"Accuracy: "
    f"{results_df['accuracy'].mean():.4f}"
)


print(
    f"Threat Recall: "
    f"{results_df['threat_recall'].mean():.4f}"
)


print(
    f"Threat F1: "
    f"{results_df['threat_f1'].mean():.4f}"
)


print(
    f"Macro F1: "
    f"{results_df['macro_f1'].mean():.4f}"
)


print(
    f"False Alarm Rate: "
    f"{results_df['false_alarm_rate'].mean():.4f}"
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


print(
    f"\nResults saved to:\n"
    f"{OUTPUT_PATH}"
)


print("\n" + "=" * 95)
print("V2 CHARACTER TEST COMPLETE")
print("=" * 95)