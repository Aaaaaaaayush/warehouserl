/**
 * warehouse.js — Warehouse Trajectory Visualizer Panel
 * Features 70% canvas grid, playback controls, stage selector, and live agent legend.
 */

import { api } from '../core/api.js';
import { WarehouseGrid } from '../components/WarehouseGrid.js';
import { AgentLegend } from '../components/AgentLegend.js';
import { EpisodePlayer } from '../components/EpisodePlayer.js';

let _grid = null;
let _legend = null;
let _player = null;

export async function render() {
  return `
    <div style="display: flex; gap: var(--space-4); height: calc(100vh - var(--topbar-h) - var(--space-4) * 2);">
      
      <!-- Left Controls Panel (30% width) -->
      <div style="width: 320px; display: flex; flex-direction: column; gap: var(--space-3); background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-lg); padding: var(--space-3);">
        
        <div style="font-family: var(--font-mono); font-weight: var(--weight-bold); color: var(--color-data); border-bottom: 1px solid var(--color-border); padding-bottom: 8px;">
          EPISODE VISUALIZER
        </div>

        <!-- Scenario & Stage Selectors -->
        <div style="display: flex; flex-direction: column; gap: 6px;">
          <label style="font-size: var(--text-xs); color: var(--color-muted); font-family: var(--font-mono);">SCENARIO</label>
          <select id="select-scenario" class="select">
            <option value="1">Scenario 1: Single Corridor (4 Agents)</option>
            <option value="2">Scenario 2: Open Warehouse (8 Agents)</option>
            <option value="3">Scenario 3: Full Warehouse (12 Agents)</option>
          </select>
        </div>

        <div style="display: flex; flex-direction: column; gap: 6px;">
          <label style="font-size: var(--text-xs); color: var(--color-muted); font-family: var(--font-mono);">TRAINING STAGE</label>
          <select id="select-stage" class="select">
            <option value="final" selected>Final (100% Trained)</option>
            <option value="75pct">75% Trained</option>
            <option value="50pct">50% Trained</option>
            <option value="25pct">25% Trained</option>
            <option value="episode_1">Episode 1 (Random Baseline)</option>
          </select>
        </div>

        <!-- Playback Controls -->
        <div style="display: flex; flex-direction: column; gap: 8px; border-top: 1px solid var(--color-border); border-bottom: 1px solid var(--color-border); padding: 12px 0;">
          <div style="display: flex; align-items: center; justify-content: space-between;">
            <button id="btn-play" class="btn btn-primary">▶ Play</button>
            <button id="btn-step" class="btn btn-ghost">Step ➔</button>
            <select id="select-speed" class="select" style="padding: 4px 6px;">
              <option value="0.5">0.5x</option>
              <option value="1.0" selected>1.0x</option>
              <option value="2.0">2.0x</option>
              <option value="4.0">4.0x</option>
            </select>
          </div>

          <div style="display: flex; align-items: center; justify-content: space-between; font-family: var(--font-mono); font-size: var(--text-xs); color: var(--color-muted); margin-top: 4px;">
            <span>Step: <strong id="step-counter" style="color: var(--color-data);">0</strong> / <span id="total-steps">0</span></span>
            <span>Deliveries: <strong id="delivery-counter" style="color: var(--color-success);">0</strong></span>
          </div>
        </div>

        <!-- Agent Legend Container -->
        <div id="legend-container"></div>

      </div>

      <!-- Canvas Container (70% width) -->
      <div id="canvas-container" style="flex: 1; background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-lg); display: flex; align-items: center; justify-content: center; padding: var(--space-3); overflow: hidden;">
        <div style="color: var(--color-muted); font-family: var(--font-mono);">Loading Canvas Grid...</div>
      </div>

    </div>
  `;
}

export async function afterRender() {
  const canvasContainer = document.getElementById('canvas-container');
  const legendContainer = document.getElementById('legend-container');

  const scenarioSelect = document.getElementById('select-scenario');
  const stageSelect = document.getElementById('select-stage');
  const playBtn = document.getElementById('btn-play');
  const stepBtn = document.getElementById('btn-step');
  const speedSelect = document.getElementById('select-speed');

  const stepCounter = document.getElementById('step-counter');
  const totalStepsEl = document.getElementById('total-steps');
  const deliveryCounter = document.getElementById('delivery-counter');

  _legend = new AgentLegend(legendContainer);

  async function loadTrajectory() {
    if (_player) _player.destroy();
    
    const sid = parseInt(scenarioSelect.value);
    const stage = stageSelect.value;

    let trajData = null;
    try {
      trajData = await api.getEpisode(sid, stage);
    } catch (e) {
      // Fallback mock if artifact not yet recorded
      const gridSpec = sid === 1 ? { width: 8, height: 8, shelves: [[1,1]], dispatch_points: [[3,7]], charging_stations: [[0,7]], obstacles: [] }
        : sid === 2 ? { width: 12, height: 12, shelves: [[1,1]], dispatch_points: [[5,11]], charging_stations: [[0,0]], obstacles: [] }
        : { width: 16, height: 16, shelves: [[1,1]], dispatch_points: [[7,15]], charging_stations: [[0,0]], obstacles: [] };
      
      trajData = {
        grid: gridSpec,
        total_steps: 100,
        frames: Array.from({ length: 100 }, (_, step) => ({
          step,
          agents: Array.from({ length: sid === 1 ? 4 : sid === 2 ? 8 : 12 }, (_, i) => ({
            id: `agent_${i}`, row: 2 + (i % 3), col: 2 + Math.floor(i / 3), battery: 50, carrying: step > 20 && i === 0, frozen: false
          })),
          items_on_shelves: [[1, 1]],
        }))
      };
    }

    _grid = new WarehouseGrid(canvasContainer, trajData.grid);

    _player = new EpisodePlayer(trajData, (frame, currentStep, totalSteps) => {
      _grid.updateFrame(frame);
      _legend.render(frame.agents);
      
      stepCounter.textContent = currentStep;
      totalStepsEl.textContent = totalSteps;
      
      const deliveries = frame.deliveries || 0;
      deliveryCounter.textContent = deliveries;
    });

    playBtn.textContent = '▶ Play';
  }

  scenarioSelect.addEventListener('change', loadTrajectory);
  stageSelect.addEventListener('change', loadTrajectory);

  playBtn.addEventListener('click', () => {
    _player.toggle();
    playBtn.textContent = _player.isPlaying ? '⏸ Pause' : '▶ Play';
  });

  stepBtn.addEventListener('click', () => {
    _player.pause();
    playBtn.textContent = '▶ Play';
    _player.next();
  });

  speedSelect.addEventListener('change', (e) => {
    _player.setSpeed(e.target.value);
  });

  await loadTrajectory();
}
