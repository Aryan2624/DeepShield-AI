from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["message"] == "DeepShield AI API is running"
    assert "version" in data


def test_health_endpoint():
    response = client.get("/api/health")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["data"]["status"] == "healthy"
    assert data["data"]["service"] == "DeepShield AI API"
    assert "version" in data["data"]
    assert "environment" in data["data"]