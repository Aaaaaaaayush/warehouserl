/**
 * scenarios.js — Scenarios Panel
 * Styled as a structured scientific mission briefing document.
 */

import { api } from '../core/api.js';

export async function render() {
  return `
    <div style="display: flex; flex-direction: column; gap: var(--space-4); max-width: 960px; margin: 0 auto;">
      
      <div style="border-bottom: 1px solid var(--color-border); padding-bottom: var(--space-2);">
        <h1 class="section-title">Scenario Specifications & Coordination Benchmarks</h1>
        <p style="color: var(--color-muted); font-size: var(--text-sm);">
          Curriculum learning progression from single-corridor navigation to multi-agent dynamic obstacle handling.
        </p>
      </div>

      <!-- Scenario 1 Document Block -->
      <div class="stat-card" style="padding: var(--space-4);">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: var(--space-2);">
          <h2 style="font-size: var(--text-lg); color: var(--color-accent); font-family: var(--font-mono);">Scenario 1: Single Corridor</h2>
          <span class="version-badge">100,000 EPISODES</span>
        </div>
        <p style="color: var(--color-data); font-size: var(--text-sm); margin-bottom: var(--space-3);">
          Four agents navigate an 8×8 grid connected by a single narrow corridor between shelves and the dispatch point.
          Primary coordination challenge is spatial collision avoidance and learning corridor yield protocols.
        </p>
        <table style="width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: var(--text-xs); color: var(--color-data);">
          <tr style="border-bottom: 1px solid var(--color-border);">
            <th style="text-align: left; padding: 6px; color: var(--color-muted);">GRID SIZE</th>
            <th style="text-align: left; padding: 6px; color: var(--color-muted);">AGENTS</th>
            <th style="text-align: left; padding: 6px; color: var(--color-muted);">DISPATCH POINTS</th>
            <th style="text-align: left; padding: 6px; color: var(--color-muted);">CHARGERS</th>
            <th style="text-align: left; padding: 6px; color: var(--color-muted);">WEIGHT INIT</th>
          </tr>
          <tr>
            <td style="padding: 6px;">8×8 (64 cells)</td>
            <td style="padding: 6px;">4 Robots</td>
            <td style="padding: 6px;">1 Point</td>
            <td style="padding: 6px;">2 Stations</td>
            <td style="padding: 6px;">Random (Scratch)</td>
          </tr>
        </table>
      </div>

      <!-- Scenario 2 Document Block -->
      <div class="stat-card" style="padding: var(--space-4);">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: var(--space-2);">
          <h2 style="font-size: var(--text-lg); color: var(--color-accent); font-family: var(--font-mono);">Scenario 2: Open Warehouse</h2>
          <span class="version-badge">300,000 EPISODES</span>
        </div>
        <p style="color: var(--color-data); font-size: var(--text-sm); margin-bottom: var(--space-3);">
          Eight agents in a 12×12 grid with central shelf obstacle clusters. Dual dispatch points require agents to balance traffic and route dynamically without swarming a single point.
        </p>
        <table style="width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: var(--text-xs); color: var(--color-data);">
          <tr style="border-bottom: 1px solid var(--color-border);">
            <th style="text-align: left; padding: 6px; color: var(--color-muted);">GRID SIZE</th>
            <th style="text-align: left; padding: 6px; color: var(--color-muted);">AGENTS</th>
            <th style="text-align: left; padding: 6px; color: var(--color-muted);">DISPATCH POINTS</th>
            <th style="text-align: left; padding: 6px; color: var(--color-muted);">CHARGERS</th>
            <th style="text-align: left; padding: 6px; color: var(--color-muted);">WEIGHT INIT</th>
          </tr>
          <tr>
            <td style="padding: 6px;">12×12 (144 cells)</td>
            <td style="padding: 6px;">8 Robots</td>
            <td style="padding: 6px;">2 Points</td>
            <td style="padding: 6px;">4 Stations</td>
            <td style="padding: 6px;">Scenario 1 Weights</td>
          </tr>
        </table>
      </div>

      <!-- Scenario 3 Document Block -->
      <div class="stat-card" style="padding: var(--space-4);">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: var(--space-2);">
          <h2 style="font-size: var(--text-lg); color: var(--color-accent); font-family: var(--font-mono);">Scenario 3: Full Warehouse</h2>
          <span class="version-badge">500,000 EPISODES</span>
        </div>
        <p style="color: var(--color-data); font-size: var(--text-sm); margin-bottom: var(--space-3);">
          Twelve agents in a 16×16 grid with randomized obstacle layouts per episode reset. Forces agents to abandon fixed paths and develop adaptive role specialisation under continuous item spawning.
        </p>
        <table style="width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: var(--text-xs); color: var(--color-data);">
          <tr style="border-bottom: 1px solid var(--color-border);">
            <th style="text-align: left; padding: 6px; color: var(--color-muted);">GRID SIZE</th>
            <th style="text-align: left; padding: 6px; color: var(--color-muted);">AGENTS</th>
            <th style="text-align: left; padding: 6px; color: var(--color-muted);">DISPATCH POINTS</th>
            <th style="text-align: left; padding: 6px; color: var(--color-muted);">CHARGERS</th>
            <th style="text-align: left; padding: 6px; color: var(--color-muted);">WEIGHT INIT</th>
          </tr>
          <tr>
            <td style="padding: 6px;">16×16 (256 cells)</td>
            <td style="padding: 6px;">12 Robots</td>
            <td style="padding: 6px;">4 Points</td>
            <td style="padding: 6px;">6 Stations</td>
            <td style="padding: 6px;">Scenario 2 Weights</td>
          </tr>
        </table>
      </div>

    </div>
  `;
}
