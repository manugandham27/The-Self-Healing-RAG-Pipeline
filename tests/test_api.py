"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_api_health():
    """Test /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_api_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_api_query():
    """Test /query endpoint returning pipeline response."""
    payload = {
        "query": "What encryption standard is required for data in transit?",
        "max_retries": 1,
        "enable_self_healing": True
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "status" in data
    assert "traces" in data
