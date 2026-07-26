/**
 * architecture.js — QMIX Architecture & Neural Network Spec Panel
 * Features CSS-animated network data flow diagram and parameter specification tables.
 */

export async function render() {
  return `
    <div style="display: flex; flex-direction: column; gap: var(--space-4); max-width: 960px; margin: 0 auto;">
      
      <div style="border-bottom: 1px solid var(--color-border); padding-bottom: var(--space-2);">
        <h1 class="section-title">QMIX Architecture & Centralised Training Specification</h1>
        <p style="color: var(--color-muted); font-size: var(--text-sm);">
          Centralised Training with Decentralised Execution (CTDE). Individual agent GRU networks feed monotonic mixing hypernetworks.
        </p>
      </div>

      <!-- CSS Animated Data Flow Diagram -->
      <div class="stat-card" style="padding: var(--space-4); background: var(--color-surface); text-align: center;">
        <div style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--color-muted); margin-bottom: var(--space-3);">
          DATA FLOW DIAGRAM (CTDE PRINCIPLE)
        </div>

        <div style="display: flex; align-items: center; justify-content: space-around; gap: var(--space-2); flex-wrap: wrap;">
          
          <!-- Box 1: Local Observations -->
          <div style="border: 1px solid var(--color-border); background: var(--color-bg); padding: var(--space-2); border-radius: var(--radius-md); min-width: 140px;">
            <div style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--color-accent); font-weight: bold;">Local Obs (5x5)</div>
            <div style="font-size: var(--text-xs); color: var(--color-muted); margin-top: 4px;">Vector: (31,)</div>
          </div>

          <div style="color: var(--color-accent); font-size: var(--text-xl);">➔</div>

          <!-- Box 2: Individual Q-Networks -->
          <div style="border: 1px solid var(--color-accent); background: var(--color-accent-10); padding: var(--space-2); border-radius: var(--radius-md); min-width: 160px;">
            <div style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--color-data); font-weight: bold;">Agent Q-Networks</div>
            <div style="font-size: var(--text-xs); color: var(--color-muted); margin-top: 4px;">FC(64) + GRU(64)</div>
            <div style="font-size: 10px; color: var(--color-accent); margin-top: 2px;">Parameter Sharing</div>
          </div>

          <div style="color: var(--color-accent); font-size: var(--text-xl);">➔</div>

          <!-- Box 3: Individual Q-Values -->
          <div style="border: 1px solid var(--color-border); background: var(--color-bg); padding: var(--space-2); border-radius: var(--radius-md); min-width: 120px;">
            <div style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--color-data); font-weight: bold;">Q_i (1..N)</div>
            <div style="font-size: var(--text-xs); color: var(--color-muted); margin-top: 4px;">Max Q per Agent</div>
          </div>

          <div style="color: var(--color-accent); font-size: var(--text-xl);">➔</div>

          <!-- Box 4: QMIX Mixing Network -->
          <div style="border: 1px solid var(--color-success); background: var(--color-success-10); padding: var(--space-2); border-radius: var(--radius-md); min-width: 180px;">
            <div style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--color-success); font-weight: bold;">QMIX Mixer (Monotone)</div>
            <div style="font-size: var(--text-xs); color: var(--color-muted); margin-top: 4px;">State-Conditioned w ≥ 0</div>
            <div style="font-size: 10px; color: var(--color-success); margin-top: 2px;">Hypernetworks w1, w2</div>
          </div>

          <div style="color: var(--color-success); font-size: var(--text-xl);">➔</div>

          <!-- Box 5: Q_tot -->
          <div style="border: 1px solid var(--color-success); background: var(--color-bg); padding: var(--space-2); border-radius: var(--radius-md); min-width: 100px;">
            <div style="font-family: var(--font-mono); font-size: var(--text-sm); color: var(--color-success); font-weight: bold;">Q_tot</div>
            <div style="font-size: 10px; color: var(--color-muted); margin-top: 4px;">Joint Value</div>
          </div>

        </div>
      </div>

      <!-- Parameter Table -->
      <div class="stat-card" style="padding: var(--space-4);">
        <div style="font-family: var(--font-mono); font-size: var(--text-sm); font-weight: bold; color: var(--color-data); margin-bottom: var(--space-2);">
          NETWORK ARCHITECTURE SPECIFICATION
        </div>
        <table style="width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: var(--text-xs); color: var(--color-data);">
          <thead>
            <tr style="border-bottom: 1px solid var(--color-border); text-align: left;">
              <th style="padding: 8px; color: var(--color-muted);">COMPONENT</th>
              <th style="padding: 8px; color: var(--color-muted);">LAYER TYPE</th>
              <th style="padding: 8px; color: var(--color-muted);">INPUT SHAPE</th>
              <th style="padding: 8px; color: var(--color-muted);">OUTPUT SHAPE</th>
              <th style="padding: 8px; color: var(--color-muted);">ACTIVATION</th>
              <th style="padding: 8px; color: var(--color-muted);">PARAMETERS</th>
            </tr>
          </thead>
          <tbody>
            <tr style="border-bottom: 1px dashed var(--color-border);">
              <td style="padding: 8px;">Agent QNetwork</td>
              <td style="padding: 8px;">Linear Input FC</td>
              <td style="padding: 8px;">(31,)</td>
              <td style="padding: 8px;">(64,)</td>
              <td style="padding: 8px;">ReLU</td>
              <td style="padding: 8px;">2,048</td>
            </tr>
            <tr style="border-bottom: 1px dashed var(--color-border);">
              <td style="padding: 8px;">Agent QNetwork</td>
              <td style="padding: 8px;">Recurrent GRU</td>
              <td style="padding: 8px;">(64,)</td>
              <td style="padding: 8px;">(64,)</td>
              <td style="padding: 8px;">Tanh / Sigmoid</td>
              <td style="padding: 8px;">24,960</td>
            </tr>
            <tr style="border-bottom: 1px dashed var(--color-border);">
              <td style="padding: 8px;">Agent QNetwork</td>
              <td style="padding: 8px;">Linear Output FC</td>
              <td style="padding: 8px;">(64,)</td>
              <td style="padding: 8px;">(7,) [Actions]</td>
              <td style="padding: 8px;">Linear</td>
              <td style="padding: 8px;">455</td>
            </tr>
            <tr style="border-bottom: 1px dashed var(--color-border);">
              <td style="padding: 8px;">QMIX Mixer</td>
              <td style="padding: 8px;">Hypernet w1</td>
              <td style="padding: 8px;">Global State S</td>
              <td style="padding: 8px;">(N, 32)</td>
              <td style="padding: 8px;">|Linear| (abs)</td>
              <td style="padding: 8px;">~12,500</td>
            </tr>
            <tr>
              <td style="padding: 8px;">QMIX Mixer</td>
              <td style="padding: 8px;">Hypernet w2</td>
              <td style="padding: 8px;">Global State S</td>
              <td style="padding: 8px;">(32, 1)</td>
              <td style="padding: 8px;">|Linear| (abs)</td>
              <td style="padding: 8px;">~6,200</td>
            </tr>
          </tbody>
        </table>
      </div>

    </div>
  `;
}
