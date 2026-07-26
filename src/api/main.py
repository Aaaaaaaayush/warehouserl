"""
main.py
-------
FastAPI application entry point for WarehouseRL.

WHY FASTAPI:
  FastAPI is a modern Python web framework that auto-generates
  OpenAPI documentation, has async support for I/O-bound operations
  (reading JSON files, serving static assets), and is extremely fast
  for a serving-only workload (no model inference in V1).

V1 DESIGN PRINCIPLE:
  This server does NOT load any model weights. It serves only
  pre-computed artifacts: trajectory JSONs, MLflow metric exports,
  behavior detection results, and heatmap images. This keeps the
  Oracle VM CPU-only and the latency near-zero (<5ms).

[V2-READY]: WebSocket manager and live inference worker imported but
  not activated in V1. The /ws/live endpoint returns 501.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.api.ws_stubs import ws_router

# Application metadata (appears in auto-generated /docs)
app = FastAPI(
    title="WarehouseRL API",
    description="Cooperative MARL artifact server — QMIX | PettingZoo | RTX 5080",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS: allow frontend origin (localhost dev + Oracle production domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Register HTTP API endpoints under /api
app.include_router(router, prefix="/api")

# Register WebSocket endpoints under /ws [V2-READY]
app.include_router(ws_router)

# Mount logs directory to serve heatmap PNG images at /logs/heatmap_scenario_X.png
_logs_dir = Path(__file__).parent.parent.parent / "logs"
_logs_dir.mkdir(exist_ok=True)
app.mount("/logs", StaticFiles(directory=str(_logs_dir)), name="logs")

# Mount videos directory to serve MP4 recordings at /videos/scenario_X_stage.mp4
_videos_dir = Path(__file__).parent.parent.parent / "videos"
_videos_dir.mkdir(exist_ok=True)
app.mount("/videos", StaticFiles(directory=str(_videos_dir)), name="videos")

# Serve the frontend as static files at the root URL
_frontend_dir = Path(__file__).parent.parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
