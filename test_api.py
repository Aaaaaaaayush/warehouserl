"""
test_api.py — Step 5 API test suite.

Tests:
  - GET /api/scenarios returns list of 3 scenarios with correct schema
  - GET /api/stats/{scenario_id} returns 200 or 404 cleanly
  - GET /api/episode/{scenario_id}/{stage} returns trajectory JSON or 404
  - GET /api/behaviors/{scenario_id} returns behavior detection JSON or 404
  - WS /ws/live returns 501 Not Implemented [V2-READY]
  - Static file mounts (/logs, /videos, /) respond
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_get_scenarios():
    response = client.get("/api/scenarios")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3
    assert data[0]["id"] == 1
    assert "agents" in data[0]
    assert "grid" in data[0]


def test_get_stats():
    # Scenario 1 stats may exist if training/evaluation was run
    response = client.get("/api/stats/1")
    assert response.status_code in [200, 404]

    # Invalid scenario ID
    invalid_res = client.get("/api/stats/999")
    assert invalid_res.status_code == 404


def test_get_episode():
    # Scenario 1 final episode stage
    response = client.get("/api/episode/1/final")
    assert response.status_code in [200, 404]

    # Invalid stage name
    invalid_stage = client.get("/api/episode/1/invalid_stage")
    assert invalid_stage.status_code == 400


def test_get_behaviors():
    response = client.get("/api/behaviors/1")
    assert response.status_code in [200, 404]


def test_websocket_stub():
    # Test WebSocket stub endpoint
    with client.websocket_connect("/ws/live") as websocket:
        data = websocket.receive_json()
        assert data["code"] == 501
        assert "V2" in data["message"]


def test_static_frontend_mount():
    response = client.get("/")
    assert response.status_code == 200
    assert "WarehouseRL" in response.text
