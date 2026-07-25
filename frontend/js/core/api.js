/**
 * api.js — API client for WarehouseRL backend
 *
 * WHY THIS EXISTS:
 *   Centralises all backend communication. Panels never call fetch()
 *   directly — they call api.getStats(1) etc. This means:
 *   - Base URL is defined once (easy to change for deployment)
 *   - Error handling is consistent across all panels
 *   - Caching is trivially added here without touching panel code
 *
 * [V2-READY]: WebSocket client stub is included. V2 calls
 *   api.connectLive(onFrame) to start receiving streamed frames.
 *   The WS URL mirrors the HTTP base URL automatically.
 */

import { state } from './state.js';

const _BASE = window.location.origin;   // Works locally and on Oracle
const _API  = `${_BASE}/api`;

/**
 * Internal fetch wrapper with error handling.
 * Returns parsed JSON or throws a descriptive error.
 */
async function _get(path) {
  const res = await fetch(`${_API}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status} from ${path}`);
  }
  return res.json();
}

export const api = {
  /** GET /api/scenarios — returns array of scenario metadata objects */
  async getScenarios() {
    return _get('/scenarios');
  },

  /** GET /api/stats/{id} — returns training metric time series */
  async getStats(scenarioId) {
    return _get(`/stats/${scenarioId}`);
  },

  /** GET /api/episode/{id}/{stage} — returns trajectory JSON for Canvas */
  async getEpisode(scenarioId, stage) {
    return _get(`/episode/${scenarioId}/${stage}`);
  },

  /** GET /api/behaviors/{id} — returns emergent behavior detection results */
  async getBehaviors(scenarioId) {
    return _get(`/behaviors/${scenarioId}`);
  },

  // ── [V2-READY] WebSocket live inference client ─────────────────────

  /**
   * Connect to the live inference WebSocket.
   * V1: server immediately returns 501 and closes — onError is called.
   * V2: calls onFrame(frameData) for each streamed step.
   *
   * @param {function} onFrame  Called with each streamed frame JSON
   * @param {function} onError  Called when connection fails or 501 received
   * @returns {function}        Call to disconnect
   */
  connectLive(onFrame, onError) {
    const wsUrl = _BASE.replace(/^http/, 'ws') + '/ws/live';
    let ws;
    try {
      ws = new WebSocket(wsUrl);
    } catch (e) {
      onError?.(e);
      return () => {};
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.code === 501) {
        onError?.(new Error(data.message));
        state.set('wsStatus', 'disconnected');
      } else {
        onFrame?.(data);
        state.set('wsStatus', 'streaming');
      }
    };

    ws.onerror = (e) => {
      onError?.(e);
      state.set('wsStatus', 'disconnected');
    };

    ws.onopen = () => state.set('wsStatus', 'connected');
    ws.onclose = () => state.set('wsStatus', 'disconnected');

    return () => ws.close();
  },
};
