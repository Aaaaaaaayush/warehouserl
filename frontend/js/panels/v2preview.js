/**
 * v2preview.js — Locked V2 Preview Panel
 * Displays a blurred mockup card of the interactive constraint editor with a lock icon.
 */

export async function render() {
  return `
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; min-height: 400px; text-align: center;">
      
      <div class="stat-card" style="max-width: 500px; padding: var(--space-6); position: relative; overflow: hidden; border: 1px solid var(--color-border); background: var(--color-surface);">
        
        <!-- Lock Icon Badge -->
        <div style="font-size: var(--text-3xl); margin-bottom: var(--space-2);">🔒</div>

        <h2 style="font-family: var(--font-mono); font-size: var(--text-lg); color: var(--color-data); margin-bottom: var(--space-1);">
          Interactive Constraint Editor UI
        </h2>

        <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--color-muted); margin-bottom: var(--space-4);">
          Interactive warehouse configuration — coming in V2.
        </p>

        <!-- Blurred Mockup Elements -->
        <div style="filter: blur(4px); opacity: 0.4; pointer-events: none; display: flex; flex-direction: column; gap: 8px; text-align: left; font-family: var(--font-mono); font-size: 10px;">
          <div style="background: var(--color-bg); padding: 8px; border-radius: 4px; border: 1px solid var(--color-border);">
            GRID_WIDTH: 16 | AGENT_COUNT: 12 | BATTERY_CAPACITY: 70
          </div>
          <div style="background: var(--color-bg); padding: 8px; border-radius: 4px; border: 1px solid var(--color-border);">
            COMMUNICATION_HOOKS: [ TarMAC Attention Channel ]
          </div>
          <div style="background: var(--color-bg); padding: 8px; border-radius: 4px; border: 1px solid var(--color-border);">
            LIVE_INFERENCE: [ WebSocket /ws/live Streaming ]
          </div>
        </div>

      </div>

    </div>
  `;
}
