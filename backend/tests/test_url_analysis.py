from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_google_url_analysis():
    response = client.post(
        "/api/url/analyze",
        json={
            "url": "https://www.google.com"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["url"] == "https://www.google.com"
    assert data["detector"] == "DeepShield URL Detector v3"

    assert data["status"] in ["BENIGN", "THREAT"]

    assert 0 <= data["threat_score"] <= 100
    assert 0 <= data["risk_score"] <= 100

    assert data["risk_level"] in [
        "SAFE",
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    ]

    assert "decision_threshold" in data
    assert "security_signals" in data
    assert isinstance(data["security_signals"], list)

    assert "recommended_action" in data

    assert "models" in data
    assert "stage_1" in data["models"]
    assert "stage_2" in data["models"]


def test_suspicious_url_analysis():
    suspicious_url = "http://192.168.1.20/login/verify/account"

    response = client.post(
        "/api/url/analyze",
        json={
            "url": suspicious_url
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["url"] == suspicious_url
    assert data["status"] == "THREAT"

    assert data["threat_score"] >= data["decision_threshold"]

    assert data["threat_type"] in [
        "phishing",
        "malware",
        "defacement",
    ]

    assert data["threat_type_confidence"] is not None

    assert isinstance(
        data["threat_type_scores"],
        dict,
    )

    assert len(
        data["threat_type_scores"]
    ) > 0

    assert len(
        data["security_signals"]
    ) > 0


def test_missing_url_field():
    response = client.post(
        "/api/url/analyze",
        json={},
    )

    assert response.status_code == 422


def test_invalid_request_body():
    response = client.post(
        "/api/url/analyze",
        json={
            "url": None
        },
    )

    assert response.status_code == 422