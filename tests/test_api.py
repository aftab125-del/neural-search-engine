import pytest
from starlette.testclient import TestClient
from neuralsearch.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "bi_encoder" in data["models"]


def test_ui_serves_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "NEURAL" in res.text
    assert "Warm Alabaster" in res.text or "Local-First" in res.text


def test_search_api(client):
    res = client.get("/api/search?q=neural+search+architecture&mode=hybrid")
    assert res.status_code == 200
    data = res.json()
    assert "hits" in data
    assert "telemetry" in data
    assert "total_ms" in data["telemetry"]


def test_stats_api(client):
    res = client.get("/api/stats")
    assert res.status_code == 200
    data = res.json()
    assert "total_documents" in data
    assert "total_chunks" in data
