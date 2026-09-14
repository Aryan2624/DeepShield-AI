from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


DATASET_DIR = Path(
    r"E:\DeepShield-Datasets\phishing_messages\processed"
)

MODEL_PATH = Path(
    r"C:\Users\dubey\DeepShield-AI\backend\ml\saved_models\phishing_detector_v1_candidate.joblib"
)

OUTPUT_PATH = Path(
    r"C:\Users\dubey\DeepShield-AI\backend\ml\phishing_detection\source_validation_v1.csv"
)


VALIDATION_PATH = (
    DATASET_DIR
    / "phishing_validation_v1.csv"
)


print("=" * 90)
print("DEEPSHIELD AI - SOURCE-WISE VALIDATION AUDIT")
print("=" * 90)


# ============================================================
# LOAD MODEL
# ============================================================

artifact = joblib.load(
    MODEL_PATH
)

vectorizer = artifact[
    "vectorizer"
]

model = artifact[
    "model"
]


print(
    f"\nModel: "
    f"{artifact['model_name']}"
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

validation_df = pd.read_csv(
    VALIDATION_PATH,
    low_memory=False,
)

validation_df["text"] = (
    validation_df["text"]
    .fillna("")
    .astype(str)
)


print(
    f"Validation samples: "
    f"{len(validation_df):,}"
)


# ============================================================
# TRANSFORM ONCE
# ============================================================

X_validation = vectorizer.transform(
    validation_df["text"]
)

predictions = model.predict(
    X_validation
)


validation_df[
    "prediction"
] = predictions


# ============================================================
# SOURCE-WISE ANALYSIS
# ============================================================

results = []


for source in sorted(
    validation_df["source"].unique()
):

    source_df = validation_df[
        validation_df["source"] == source
    ].copy()


    y_true = source_df[
        "label"
    ].astype(int)

    y_pred = source_df[
        "prediction"
    ].astype(int)


    accuracy = accuracy_score(
        y_true,
        y_pred,
    )


    legitimate_count = int(
        (y_true == 0).sum()
    )

    threat_count = int(
        (y_true == 1).sum()
    )


    if threat_count > 0:

        threat_precision = precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        )

        threat_recall = recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        )

        threat_f1 = f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        )

    else:

        threat_precision = None
        threat_recall = None
        threat_f1 = None


    if legitimate_count > 0:

        legitimate_recall = recall_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0,
        )

    else:

        legitimate_recall = None


    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )


    false_alarm_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else None
    )


    results.append(
        {
            "source": source,
            "samples": len(source_df),
            "legitimate_samples": legitimate_count,
            "threat_samples": threat_count,
            "accuracy": accuracy,
            "legitimate_recall": legitimate_recall,
            "threat_precision": threat_precision,
            "threat_recall": threat_recall,
            "threat_f1": threat_f1,
            "false_alarm_rate": false_alarm_rate,
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }
    )


results_df = pd.DataFrame(
    results
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 90)
print("SOURCE-WISE RESULTS")
print("=" * 90)


for _, row in results_df.iterrows():

    print("\n" + "-" * 90)

    print(
        f"Source: {row['source']}"
    )

    print(
        f"Samples: "
        f"{int(row['samples']):,}"
    )

    print(
        f"Legitimate: "
        f"{int(row['legitimate_samples']):,}"
    )

    print(
        f"Threat: "
        f"{int(row['threat_samples']):,}"
    )

    print(
        f"Accuracy: "
        f"{row['accuracy']:.4f}"
    )


    if pd.notna(
        row["legitimate_recall"]
    ):
        print(
            f"Legitimate Recall: "
            f"{row['legitimate_recall']:.4f}"
        )


    if pd.notna(
        row["threat_precision"]
    ):
        print(
            f"Threat Precision: "
            f"{row['threat_precision']:.4f}"
        )

        print(
            f"Threat Recall: "
            f"{row['threat_recall']:.4f}"
        )

        print(
            f"Threat F1: "
            f"{row['threat_f1']:.4f}"
        )


    if pd.notna(
        row["false_alarm_rate"]
    ):
        print(
            f"False Alarm Rate: "
            f"{row['false_alarm_rate']:.4f}"
        )


    print(
        "Confusion:"
    )

    print(
        f"TN={int(row['true_negative'])}, "
        f"FP={int(row['false_positive'])}, "
        f"FN={int(row['false_negative'])}, "
        f"TP={int(row['true_positive'])}"
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


print("\n" + "=" * 90)
print("AUDIT SUMMARY")
print("=" * 90)


weak_sources = results_df[
    results_df["accuracy"] < 0.90
]


print(
    f"Sources below 90% accuracy: "
    f"{len(weak_sources)}"
)


if len(weak_sources) > 0:

    print(
        weak_sources[
            [
                "source",
                "accuracy",
            ]
        ].to_string(
            index=False
        )
    )


print(
    f"\nResults saved to:\n"
    f"{OUTPUT_PATH}"
)


print("\n" + "=" * 90)
print("SOURCE AUDIT COMPLETE")
print("=" * 90)