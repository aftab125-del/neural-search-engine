"""
Unit tests verifying Vercel serverless entrypoint and behavior.
"""
import os
import sys
from pathlib import Path
from starlette.testclient import TestClient

def test_vercel_entrypoint_import():
    """Verify that api/index.py can be imported and exposes the FastAPI app."""
    # Ensure api directory is on path
    api_dir = Path(__file__).resolve().parent.parent / "api"
    if str(api_dir) not in sys.path:
        sys.path.insert(0, str(api_dir))

    import index
    assert hasattr(index, "app")
    assert index.app.title == "NeuralSearch API"

def test_vercel_endpoints_and_health():
    """Verify health endpoint reports correct serverless and model state."""
    import index
    client = TestClient(index.app)

    # 1. UI serves HTML
    resp = client.get("/")
    assert resp.status_code == 200
    assert "NeuralSearch" in resp.text
    assert "<!DOCTYPE html>" in resp.text

    # 2. Health endpoint
    h_resp = client.get("/api/health")
    assert h_resp.status_code == 200
    data = h_resp.json()
    assert data["status"] == "healthy"
    assert "all" in data["features"]
    assert "photo_search" in data["features"]
    assert "serverless" in data

    # 3. Autocomplete
    ac_resp = client.get("/api/autocomplete?q=python")
    assert ac_resp.status_code == 200
    assert "suggestions" in ac_resp.json()
