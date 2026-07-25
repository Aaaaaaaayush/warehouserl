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
  Oracle VM CPU-only and the latency near-zero.

[V2-READY]: WebSocket manager and live inference worker imported but
  not activated in V1. The /ws/live endpoint returns 501.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from src.api.routes import router

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
    allow_origins=["*"],   # Tighten in production if needed
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Register all HTTP + WebSocket routes from routes.py
app.include_router(router, prefix="/api")

# Serve the frontend as static files at the root URL
_frontend_dir = Path(__file__).parent.parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
