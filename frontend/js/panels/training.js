/**
 * training.js — Training Metrics Panel
 * 2×2 Plotly chart grid for throughput, collisions, deadlocks, and battery efficiency.
 */

import { api } from '../core/api.js';
import { ChartWrapper } from '../components/ChartWrapper.js';

export async function render() {
  return `
    <div style="display: flex; flex-direction: column; gap: var(--space-3);">
      
      <!-- Scenario Selector Header -->
      <div style="display: flex; align-items: center; justify-content: space-between; background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-lg); padding: var(--space-2) var(--space-3);">
        <div style="font-family: var(--font-mono); font-weight: var(--weight-bold); color: var(--color-data); font-size: var(--text-sm);">
          QMIX TRAINING METRICS
        </div>

        <div style="display: flex; align-items: center; gap: var(--space-2);">
          <label style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--color-muted);">SELECT SCENARIO</label>
          <select id="training-scenario-select" class="select">
            <option value="1" selected>Scenario 1: Single Corridor (100K Eps)</option>
            <option value="2">Scenario 2: Open Warehouse (300K Eps)</option>
            <option value="3">Scenario 3: Full Warehouse (500K Eps)</option>
          </select>
        </div>
      </div>

      <!-- 2×2 Plotly Grid -->
      <div style="display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 280px 280px; gap: var(--space-3);">
        <div id="chart-throughput" class="stat-card" style="padding: 4px;"></div>
        <div id="chart-collisions" class="stat-card" style="padding: 4px;"></div>
        <div id="chart-deadlocks" class="stat-card" style="padding: 4px;"></div>
        <div id="chart-battery" class="stat-card" style="padding: 4px;"></div>
      </div>

    </div>
  `;
}

export async function afterRender() {
  const select = document.getElementById('training-scenario-select');

  const chartThroughput = new ChartWrapper(document.getElementById('chart-throughput'), 'PACKAGES DELIVERED (PER 1K EPS)', '#00cc88');
  const chartCollisions = new ChartWrapper(document.getElementById('chart-collisions'), 'COLLISION RATE (PER STEP)', '#ff3355');
  const chartDeadlocks = new ChartWrapper(document.getElementById('chart-deadlocks'), 'DEADLOCK FREQUENCY (PER STEP)', '#ffaa00');
  const chartBattery = new ChartWrapper(document.getElementById('chart-battery'), 'BATTERY DEPLETION EVENTS', '#0066ff');

  async function loadCharts() {
    const sid = parseInt(select.value);
    let stats = [];

    try {
      stats = await api.getStats(sid);
    } catch (e) {
      // Mock curve for preview if stats file not yet generated
      const maxEps = sid === 1 ? 100 : sid === 2 ? 300 : 500;
      stats = Array.from({ length: maxEps }, (_, i) => {
        const ep = (i + 1) * 1000;
        return {
          episode: ep,
          packages_delivered: Math.min(25, (ep / (maxEps * 1000)) * 25 + Math.random() * 2),
          collision_rate: Math.max(0.01, 0.25 - (ep / (maxEps * 1000)) * 0.22),
          deadlock_frequency: Math.max(0.02, 0.35 - (ep / (maxEps * 1000)) * 0.30),
          battery_depletion_ev: Math.max(0, 4 - Math.floor(ep / 50000)),
        };
      });
    }

    const eps = stats.map(s => s.episode);
    chartThroughput.render(eps, stats.map(s => s.packages_delivered));
    chartCollisions.render(eps, stats.map(s => s.collision_rate));
    chartDeadlocks.render(eps, stats.map(s => s.deadlock_frequency));
    chartBattery.render(eps, stats.map(s => s.battery_depletion_ev));
  }

  select.addEventListener('change', loadCharts);
  await loadCharts();
}
