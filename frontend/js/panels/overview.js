/**
 * overview.js — Overview Panel
 * Displays total packages hero metric and scenario summary cards.
 */

import { api } from '../core/api.js';

export async function render() {
  let scenarios = [];
  try {
    scenarios = await api.getScenarios();
  } catch (e) {
    scenarios = [
      { id: 1, name: "Single Corridor", agents: 4, grid: "8×8", episodes: 100000, challenge: "Basic navigation and collision avoidance" },
      { id: 2, name: "Open Warehouse", agents: 8, grid: "12×12", episodes: 300000, challenge: "Traffic management & lane formation" },
      { id: 3, name: "Full Warehouse", agents: 12, grid: "16×16", episodes: 500000, challenge: "Role specialisation under dynamic conditions" },
    ];
  }

  return `
    <div style="display: flex; flex-direction: column; gap: var(--space-4);">
      
      <!-- Hero Metric Header -->
      <div class="stat-card" style="align-items: center; text-align: center; padding: var(--space-6) var(--space-4); background: var(--color-surface); border: 1px solid var(--color-border);">
        <div class="hero-metric" id="hero-delivered-count">900,000</div>
        <div class="hero-label">Total Training Episodes Across 3 Scenarios</div>
        <div style="margin-top: var(--space-2); color: var(--color-muted); font-size: var(--text-xs); font-family: var(--font-mono);">
          Cooperative MARL · QMIX Algorithm · PyTorch 2.10 · NVIDIA RTX 5080
        </div>
      </div>

      <div class="accent-rule">SCENARIO SUMMARY</div>

      <!-- 3 Scenario Cards in a Row -->
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: var(--space-3);">
        ${scenarios.map(s => `
          <div class="stat-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
              <span class="stat-card-label">SCENARIO ${s.id}</span>
              <span class="version-badge">${s.agents} AGENTS</span>
            </div>
            <div style="font-size: var(--text-lg); font-weight: var(--weight-bold); color: var(--color-data); margin: var(--space-1) 0;">
              ${s.name}
            </div>
            <div style="font-size: var(--text-xs); color: var(--color-muted); font-family: var(--font-mono);">
              Grid: ${s.grid} · Budget: ${s.episodes.toLocaleString()} eps
            </div>
            <div style="margin-top: var(--space-2); font-size: var(--text-xs); color: var(--color-data); border-top: 1px dashed var(--color-border); padding-top: var(--space-2);">
              ${s.challenge}
            </div>
          </div>
        `).join('')}
      </div>

    </div>
  `;
}
