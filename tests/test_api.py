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
    assert "Private & Ad-Free" in res.text or "Local-First" in res.text


def test_search_api_hybrid(client):
    res = client.get("/api/search?q=neural+search+architecture&mode=hybrid")
    assert res.status_code == 200
    data = res.json()
    assert "hits" in data
    assert "telemetry" in data
    assert "total_ms" in data["telemetry"]


def test_search_api_web(client):
    res = client.get("/api/search?q=fastapi+tutorial&mode=web&top_k=5")
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "web"
    assert "hits" in data
    assert "ads_blocked_count" in data
    assert "trackers_purged_count" in data
    assert "telemetry" in data


def test_stats_api(client):
    res = client.get("/api/stats")
    assert res.status_code == 200
    data = res.json()
    assert "total_documents" in data
    assert "total_chunks" in data
