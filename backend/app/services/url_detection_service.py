from ml.url_detection.predict import predict_url


def analyze_url(url: str) -> dict:
    """
    Analyze a URL using the DeepShield URL Detector v3 pipeline.

    Stage 1:
        BENIGN vs THREAT

    Stage 2:
        PHISHING / MALWARE / DEFACEMENT
        (only when Stage 1 flags a threat)
    """

    return predict_url(url)