"""
ws_stubs.py
-----------
WebSocket endpoint stubs — V2-ready placeholders.

WHY THIS EXISTS NOW:
  The WebSocket route /ws/live is registered in routes.py and returns
  HTTP 501 Not Implemented. This is intentional:
  - It documents that V2 will add real-time inference streaming
  - Frontend api.js already has the WS client stub waiting
  - V2 only needs to replace this stub with real logic

[V2-READY]: Replace the 501 stub below with:
  - Load model weights from latest checkpoint
  - Accept episode control messages from client (reset, step, speed)
  - Stream frame data at 20 FPS as JSON events
  - Support multi-user sessions with isolated env instances
"""

from fastapi import WebSocket, WebSocketDisconnect
from fastapi import APIRouter

ws_router = APIRouter()


@ws_router.websocket("/ws/live")
async def live_inference_stub(websocket: WebSocket):
    """
    [V2-READY] Live inference WebSocket.

    V1: Accepts the connection, immediately sends a 501 message,
        then closes cleanly. The frontend handles this gracefully
        by showing the "V2 Preview" locked panel.

    V2: Will stream real-time agent frames as JSON events at 20 FPS.
    """
    await websocket.accept()
    await websocket.send_json({
        "type": "error",
        "code": 501,
        "message": "Live inference not available in V1. Coming in V2.",
        "v2_eta": "roadmap",
    })
    await websocket.close(code=1000)
