"""
routes.py
---------
HTTP API endpoints for WarehouseRL.

All endpoints serve pre-computed artifacts from disk.
No model inference occurs in V1.

Endpoints:
  GET  /api/scenarios             — scenario metadata for all 3 scenarios
  GET  /api/stats/{scenario_id}   — MLflow training metrics as JSON
  GET  /api/episode/{scenario_id}/{stage} — trajectory JSON for frontend Canvas
  GET  /api/behaviors/{scenario_id}       — detected behaviors + heatmap paths

WebSocket:
  WS /ws/live — stub, returns 501 Not Implemented [V2-READY]
  (implemented in ws_stubs.py)
"""

from __future__ import annotations

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

_EPISODES_DIR = Path("episodes")
_LOGS_DIR     = Path("logs")

# ── Scenario metadata ─────────────────────────────────────────────────────────

_SCENARIO_META = {
    1: {
        "id": 1, "name": "Single Corridor",
        "agents": 4, "grid": "8×8",
        "episodes": 100_000,
        "challenge": "Basic navigation and collision avoidance in a tight layout",
    },
    2: {
        "id": 2, "name": "Open Warehouse",
        "agents": 8, "grid": "12×12",
        "episodes": 300_000,
        "challenge": "Traffic management, lane formation, multi-dispatch routing",
    },
    3: {
        "id": 3, "name": "Full Warehouse",
        "agents": 12, "grid": "16×16",
        "episodes": 500_000,
        "challenge": "Role specialisation under randomised obstacle layouts",
    },
}


@router.get("/scenarios")
async def get_scenarios():
    """Return metadata for all three training scenarios."""
    return JSONResponse(content=list(_SCENARIO_META.values()))


@router.get("/stats/{scenario_id}")
async def get_stats(scenario_id: int):
    """
    Return MLflow training metrics for a scenario as JSON.
    Reads from logs/scenario_{id}_metrics.json exported during training.
    """
    if scenario_id not in _SCENARIO_META:
        raise HTTPException(status_code=404, detail=f"Scenario {scenario_id} not found")

    metrics_path = _LOGS_DIR / f"scenario_{scenario_id}_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Metrics not yet generated. Run training first."
        )

    with metrics_path.open() as f:
        return JSONResponse(content=json.load(f))


@router.get("/episode/{scenario_id}/{stage}")
async def get_episode(scenario_id: int, stage: str):
    """
    Return a recorded trajectory JSON for the frontend Canvas player.
    stage: "episode_1" | "25pct" | "50pct" | "75pct" | "final"
    [V2-READY]: Same JSON schema will be streamed live via WebSocket in V2.
    """
    if scenario_id not in _SCENARIO_META:
        raise HTTPException(status_code=404, detail=f"Scenario {scenario_id} not found")

    valid_stages = ["episode_1", "25pct", "50pct", "75pct", "final"]
    if stage not in valid_stages:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stage '{stage}'. Must be one of {valid_stages}"
        )

    ep_path = _EPISODES_DIR / f"scenario_{scenario_id}_{stage}.json"
    if not ep_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Episode not recorded yet. Run evaluation first."
        )

    with ep_path.open() as f:
        return JSONResponse(content=json.load(f))


@router.get("/behaviors/{scenario_id}")
async def get_behaviors(scenario_id: int):
    """
    Return emergent behavior detection results for a scenario.
    Includes behavior names, descriptions, first detection episode,
    frequency, and heatmap image paths.
    """
    if scenario_id not in _SCENARIO_META:
        raise HTTPException(status_code=404, detail=f"Scenario {scenario_id} not found")

    behavior_path = _LOGS_DIR / f"scenario_{scenario_id}_behaviors.json"
    if not behavior_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Behavior analysis not yet run. Execute evaluation pipeline first."
        )

    with behavior_path.open() as f:
        return JSONResponse(content=json.load(f))
