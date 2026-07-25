/**
 * state.js — Global application state store
 *
 * WHY THIS EXISTS:
 *   Multiple panels need to share data: the current scenario selection,
 *   fetched training stats, loaded trajectory, active panel name.
 *   Without a central store, each panel would independently fetch the
 *   same data, causing redundant API calls and inconsistent state.
 *
 *   This is a minimal reactive store — panels can subscribe() to
 *   specific keys and receive callbacks when they change.
 *   [V2-READY]: V2 extends this store with WebSocket-streamed live state.
 */

const _state = {
  activePanel: "overview",
  activeScenario: 1,
  scenarioMeta: null,       // Loaded from GET /api/scenarios
  stats: {},                // { 1: {...}, 2: {...}, 3: {...} }
  episodes: {},             // { "1_final": {...}, ... }
  behaviors: {},            // { 1: {...}, 2: {...}, 3: {...} }
  wsStatus: "disconnected", // [V2-READY] "connected" | "streaming" | "disconnected"
};

const _subscribers = {};   // { key: [callback, ...] }

export const state = {
  /** Read a state key. */
  get(key) {
    return _state[key];
  },

  /** Write a state key and notify subscribers. */
  set(key, value) {
    _state[key] = value;
    if (_subscribers[key]) {
      _subscribers[key].forEach(cb => cb(value));
    }
  },

  /**
   * Subscribe to changes on a specific key.
   * Returns an unsubscribe function.
   */
  subscribe(key, callback) {
    if (!_subscribers[key]) _subscribers[key] = [];
    _subscribers[key].push(callback);
    return () => {
      _subscribers[key] = _subscribers[key].filter(cb => cb !== callback);
    };
  },
};
