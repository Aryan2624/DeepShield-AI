import argparse
from pathlib import Path

import joblib
import pandas as pd

try:
    from .features import extract_url_features
except ImportError:
    from features import extract_url_features


PROJECT_ROOT = Path(__file__).resolve().parents[3]

STAGE1_MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "saved_models"
    / "url_binary_v3_stage1.joblib"
)

STAGE2_MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "saved_models"
    / "url_threat_type_v3_stage2.joblib"
)

THREAT_THRESHOLD = 0.45


def load_model(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{path}"
        )

    return joblib.load(path)


STAGE1_PACKAGE = load_model(
    STAGE1_MODEL_PATH
)

STAGE2_PACKAGE = load_model(
    STAGE2_MODEL_PATH
)

STAGE1_MODEL = STAGE1_PACKAGE["model"]
STAGE2_MODEL = STAGE2_PACKAGE["model"]

FEATURE_COLUMNS = STAGE1_PACKAGE[
    "feature_columns"
]


def get_risk_level(score):
    if score <= 24:
        return "SAFE"

    if score <= 44:
        return "LOW"

    if score <= 64:
        return "MEDIUM"

    if score <= 84:
        return "HIGH"

    return "CRITICAL"


def generate_security_signals(features):
    signals = []

    if features.get(
        "has_ip_address",
        0,
    ) == 1:
        signals.append(
            "The URL uses an IP address "
            "instead of a normal hostname."
        )

    suspicious_keywords = (
        features.get(
            "suspicious_keyword_count",
            0,
        )
    )

    if suspicious_keywords > 0:
        signals.append(
            f"The URL contains "
            f"{suspicious_keywords} "
            "suspicious keyword(s)."
        )

    if features.get(
        "shortened_url",
        0,
    ) == 1:
        signals.append(
            "The URL uses a known "
            "URL-shortening service."
        )

    if features.get(
        "suspicious_extension",
        0,
    ) == 1:
        signals.append(
            "The URL contains a potentially "
            "dangerous file extension."
        )

    if features.get(
        "has_explicit_port",
        0,
    ) == 1:
        signals.append(
            "The URL specifies an explicit "
            "network port."
        )

    encoded = features.get(
        "percent_encoding_count",
        0,
    )

    if encoded > 0:
        signals.append(
            f"The URL contains "
            f"{encoded} percent-encoded "
            "sequence(s)."
        )

    if features.get(
        "double_slash_in_path",
        0,
    ) == 1:
        signals.append(
            "The URL contains an unusual "
            "double slash in its path."
        )

    non_ascii = features.get(
        "non_ascii_count",
        0,
    )

    if non_ascii > 0:
        signals.append(
            f"The URL contains "
            f"{non_ascii} non-ASCII "
            "character(s)."
        )

    if features.get(
        "url_parse_error",
        0,
    ) == 1:
        signals.append(
            "The URL has an unusual "
            "or malformed structure."
        )

    at_count = features.get(
        "at_count",
        0,
    )

    if at_count > 0:
        signals.append(
            f"The URL contains "
            f"{at_count} '@' character(s)."
        )

    subdomains = features.get(
        "subdomain_count",
        0,
    )

    if subdomains >= 4:
        signals.append(
            f"The hostname has an unusually "
            f"high number of subdomains "
            f"({subdomains})."
        )

    hostname_digit_ratio = (
        features.get(
            "hostname_digit_ratio",
            0,
        )
    )

    if hostname_digit_ratio >= 0.30:
        signals.append(
            "A large portion of the hostname "
            "contains numeric characters."
        )

    if not signals:
        signals.append(
            "No strong individual lexical "
            "warning was triggered. The AI "
            "decision is based on the combined "
            "URL feature pattern."
        )

    return signals


def recommended_action(
    status,
    threat_type,
    risk_level,
):
    if status == "BENIGN":
        if risk_level == "SAFE":
            return (
                "No immediate threat was detected. "
                "Continue using normal browsing "
                "security practices."
            )

        return (
            "The URL was classified as benign, "
            "but some uncertainty remains. Verify "
            "the source if the link was unexpected."
        )

    if threat_type == "phishing":
        return (
            "Do not enter passwords, OTPs, "
            "banking details, or personal "
            "information. Verify the website "
            "through an independent trusted source."
        )

    if threat_type == "malware":
        return (
            "Do not open downloads or execute files "
            "from this URL. Block access and "
            "investigate the source."
        )

    if threat_type == "defacement":
        return (
            "Treat the website as potentially "
            "compromised. Avoid trusting its "
            "content until the site owner confirms "
            "that it is secure."
        )

    return (
        "Avoid accessing the URL until it has "
        "been independently investigated."
    )


def predict_url(url):
    if not isinstance(url, str):
        raise TypeError(
            "URL must be a string."
        )

    url = url.strip()

    if not url:
        raise ValueError(
            "URL cannot be empty."
        )

    features = extract_url_features(
        url
    )

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in features
    ]

    if missing_features:
        raise ValueError(
            "Missing extracted features: "
            + ", ".join(missing_features)
        )

    X = pd.DataFrame(
        [[
            features[column]
            for column in FEATURE_COLUMNS
        ]],
        columns=FEATURE_COLUMNS,
    )

    # -----------------------------------------------------
    # Stage 1 — BENIGN vs THREAT
    # -----------------------------------------------------

    stage1_probabilities = (
        STAGE1_MODEL.predict_proba(X)[0]
    )

    stage1_classes = list(
        STAGE1_MODEL.classes_
    )

    threat_index = (
        stage1_classes.index(1)
    )

    threat_score = float(
        stage1_probabilities[
            threat_index
        ]
    )

    is_threat = (
        threat_score
        >= THREAT_THRESHOLD
    )

    risk_score = int(
        round(
            threat_score * 100
        )
    )

    risk_level = get_risk_level(
        risk_score
    )

    # -----------------------------------------------------
    # Stage 2 — threat category
    # -----------------------------------------------------

    threat_type = None
    threat_type_confidence = None
    threat_type_scores = {}

    if is_threat:

        stage2_probabilities = (
            STAGE2_MODEL.predict_proba(
                X
            )[0]
        )

        stage2_classes = list(
            STAGE2_MODEL.classes_
        )

        threat_type_scores = {
            str(label):
                round(
                    float(probability)
                    * 100,
                    2,
                )
            for label, probability
            in zip(
                stage2_classes,
                stage2_probabilities,
            )
        }

        best_index = int(
            stage2_probabilities.argmax()
        )

        threat_type = str(
            stage2_classes[
                best_index
            ]
        )

        threat_type_confidence = (
            round(
                float(
                    stage2_probabilities[
                        best_index
                    ]
                )
                * 100,
                2,
            )
        )

    status = (
        "THREAT"
        if is_threat
        else "BENIGN"
    )

    signals = generate_security_signals(
        features
    )

    action = recommended_action(
        status,
        threat_type,
        risk_level,
    )

    return {
        "url": url,

        "detector": (
            "DeepShield URL Detector v3"
        ),

        "status": status,

        # Model score, not a guaranteed
        # real-world probability.
        "threat_score": round(
            threat_score * 100,
            2,
        ),

        "decision_threshold":
            THREAT_THRESHOLD * 100,

        "risk_score":
            risk_score,

        "risk_level":
            risk_level,

        "threat_type":
            threat_type,

        "threat_type_confidence":
            threat_type_confidence,

        "threat_type_scores":
            threat_type_scores,

        "security_signals":
            signals,

        "recommended_action":
            action,

        "models": {
            "stage_1":
                STAGE1_PACKAGE[
                    "model_name"
                ],

            "stage_2":
                STAGE2_PACKAGE[
                    "model_name"
                ],
        },
    }


def print_result(result):
    print(
        "\n"
        + "=" * 74
    )

    print(
        "DeepShield AI - "
        "URL Security Analysis v3"
    )

    print(
        "=" * 74
    )

    print(
        f"\nURL              : "
        f"{result['url']}"
    )

    print(
        f"Status           : "
        f"{result['status']}"
    )

    print(
        f"Threat Score     : "
        f"{result['threat_score']:.2f}/100"
    )

    print(
        f"Decision Threshold: "
        f"{result['decision_threshold']:.0f}/100"
    )

    print(
        f"Risk Score       : "
        f"{result['risk_score']}/100"
    )

    print(
        f"Risk Level       : "
        f"{result['risk_level']}"
    )

    if (
        result["status"]
        == "THREAT"
    ):
        print(
            f"Threat Type      : "
            f"{result['threat_type'].upper()}"
        )

        print(
            f"Type Confidence  : "
            f"{result['threat_type_confidence']:.2f}%"
        )

        print(
            "\n"
            + "-" * 74
        )

        print(
            "THREAT TYPE SCORES"
        )

        print(
            "-" * 74
        )

        for (
            label,
            score,
        ) in (
            result[
                "threat_type_scores"
            ].items()
        ):
            print(
                f"{label.capitalize():12} : "
                f"{score:.2f}%"
            )

    print(
        "\n"
        + "-" * 74
    )

    print(
        "SECURITY SIGNALS"
    )

    print(
        "-" * 74
    )

    for (
        number,
        signal,
    ) in enumerate(
        result[
            "security_signals"
        ],
        start=1,
    ):
        print(
            f"{number}. {signal}"
        )

    print(
        "\n"
        + "-" * 74
    )

    print(
        "RECOMMENDED ACTION"
    )

    print(
        "-" * 74
    )

    print(
        result[
            "recommended_action"
        ]
    )

    print(
        "\n"
        + "-" * 74
    )

    print(
        "MODEL PIPELINE"
    )

    print(
        "-" * 74
    )

    print(
        "Stage 1 : "
        + result[
            "models"
        ]["stage_1"]
    )

    print(
        "Stage 2 : "
        + result[
            "models"
        ]["stage_2"]
    )

    print(
        "\n"
        + "=" * 74
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a URL using "
            "DeepShield URL Detector v3."
        )
    )

    parser.add_argument(
        "url",
        nargs="?",
        help="URL to analyze",
    )

    args = parser.parse_args()

    url = args.url

    if not url:
        url = input(
            "Enter URL to analyze: "
        ).strip()

    result = predict_url(
        url
    )

    print_result(
        result
    )


if __name__ == "__main__":
    main()