/**
 * AgentLegend.js — Agent status color key and counter display
 */

export class AgentLegend {
  constructor(container) {
    this.container = container;
  }

  render(agentStates = []) {
    const total = agentStates.length;
    const carrying = agentStates.filter(a => a.carrying).length;
    const lowBat = agentStates.filter(a => a.battery < 20 && !a.frozen).length;
    const frozen = agentStates.filter(a => a.frozen).length;

    this.container.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: var(--space-2); font-family: var(--font-mono); font-size: var(--text-xs);">
        <div style="font-weight: var(--weight-bold); color: var(--color-data); border-bottom: 1px solid var(--color-border); padding-bottom: 4px;">
          AGENT STATUS LEGEND (${total} AGENTS)
        </div>
        
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="status-dot accent"></span>
          <span style="color: var(--color-data);">Carrying Package</span>
          <span style="margin-left: auto; color: var(--color-accent); font-weight: var(--weight-bold);">${carrying}</span>
        </div>

        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="status-dot muted"></span>
          <span style="color: var(--color-data);">En-Route to Shelf</span>
          <span style="margin-left: auto; color: var(--color-muted);">${total - carrying - lowBat - frozen}</span>
        </div>

        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="status-dot warning"></span>
          <span style="color: var(--color-data);">Battery Low (&lt;20%)</span>
          <span style="margin-left: auto; color: var(--color-warning); font-weight: var(--weight-bold);">${lowBat}</span>
        </div>

        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="status-dot danger"></span>
          <span style="color: var(--color-data);">Frozen (Empty Battery)</span>
          <span style="margin-left: auto; color: var(--color-danger); font-weight: var(--weight-bold);">${frozen}</span>
        </div>
      </div>
    `;
  }
}
