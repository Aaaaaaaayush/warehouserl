/**
 * research.js — Academic Research Citations Panel
 * Typeset reference cards for QMIX and PettingZoo papers with extension notes.
 */

export async function render() {
  return `
    <div style="display: flex; flex-direction: column; gap: var(--space-4); max-width: 960px; margin: 0 auto;">
      
      <div style="border-bottom: 1px solid var(--color-border); padding-bottom: var(--space-2);">
        <h1 class="section-title">Academic Citations & Literature Foundations</h1>
        <p style="color: var(--color-muted); font-size: var(--text-sm);">
          Primary literature replicated and extended in the WarehouseRL project.
        </p>
      </div>

      <!-- Citation Card 1: QMIX -->
      <div class="stat-card" style="padding: var(--space-4);">
        <div style="display: flex; align-items: flex-start; justify-content: space-between;">
          <div>
            <div style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--color-accent); font-weight: bold;">
              PRIMARY ALGORITHM REFERENCE
            </div>
            <h2 style="font-size: var(--text-base); color: var(--color-data); margin: 4px 0;">
              QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning
            </h2>
            <div style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--color-muted);">
              Tabish Rashid, Mikayel Samvelyan, Christian Schroeder de Witt, Gregory Farquhar, Jakob Foerster, Shimon Whiteson
            </div>
            <div style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--color-accent); margin-top: 2px;">
              Proceedings of the 35th International Conference on Machine Learning (ICML 2018), PMLR 80:4295-4304.
            </div>
          </div>
          <span class="version-badge">ICML 2018</span>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-3); margin-top: var(--space-3); border-top: 1px dashed var(--color-border); padding-top: var(--space-3); font-size: var(--text-xs);">
          <div>
            <span style="font-family: var(--font-mono); color: var(--color-success); font-weight: bold;">WHAT WE REPLICATED:</span>
            <ul style="color: var(--color-muted); margin-top: 4px; padding-left: 16px; line-height: 1.6;">
              <li>Centralised Training with Decentralised Execution (CTDE) architecture.</li>
              <li>Hypernetwork-generated mixing weights with absolute non-negativity constraint ($w \ge 0$).</li>
              <li>Shared recurrent agent Q-networks (GRU) for partial observability.</li>
              <li>Replay buffer storing full episode trajectories.</li>
            </ul>
          </div>

          <div>
            <span style="font-family: var(--font-mono); color: var(--color-accent); font-weight: bold;">WHAT WE EXTENDED:</span>
            <ul style="color: var(--color-muted); margin-top: 4px; padding-left: 16px; line-height: 1.6;">
              <li>Custom reward shaping for battery resource management & dynamic shelf item spawns.</li>
              <li>Automated emergent behavior detection pipeline (Lane, Turn-taking, Specialisation, Convoy).</li>
              <li>Curriculum weight transfer across 3 scaling scenario levels.</li>
              <li>Real-time HTML5 Canvas trajectory visualization & artifact server.</li>
            </ul>
          </div>
        </div>
      </div>

      <!-- Citation Card 2: PettingZoo -->
      <div class="stat-card" style="padding: var(--space-4);">
        <div style="display: flex; align-items: flex-start; justify-content: space-between;">
          <div>
            <div style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--color-accent); font-weight: bold;">
              ENVIRONMENT FRAMEWORK REFERENCE
            </div>
            <h2 style="font-size: var(--text-base); color: var(--color-data); margin: 4px 0;">
              PettingZoo: Gym for Multi-Agent Reinforcement Learning
            </h2>
            <div style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--color-muted);">
              J. K. Terry, Benjamin Black, Nathaniel Grammel, Mario Jayakumar, Anis Hari, Ryan Sullivan, Luis S. Santos, Rodrigo Perez-Vicente, Nicholas Morrow, Caroline Horsch, Manojit Nandi
            </div>
            <div style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--color-accent); margin-top: 2px;">
              Advances in Neural Information Processing Systems 34 (NeurIPS 2021).
            </div>
          </div>
          <span class="version-badge">NeurIPS 2021</span>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-3); margin-top: var(--space-3); border-top: 1px dashed var(--color-border); padding-top: var(--space-3); font-size: var(--text-xs);">
          <div>
            <span style="font-family: var(--font-mono); color: var(--color-success); font-weight: bold;">WHAT WE REPLICATED:</span>
            <ul style="color: var(--color-muted); margin-top: 4px; padding-left: 16px; line-height: 1.6;">
              <li>Parallel API standard for simultaneous agent step execution.</li>
              <li>Strict PettingZoo environment contract compliance (`reset`, `step`, `action_space`, `observation_space`).</li>
            </ul>
          </div>

          <div>
            <span style="font-family: var(--font-mono); color: var(--color-accent); font-weight: bold;">WHAT WE EXTENDED:</span>
            <ul style="color: var(--color-muted); margin-top: 4px; padding-left: 16px; line-height: 1.6;">
              <li>Built a ground-up custom warehouse logistics grid environment (`WarehouseEnv`).</li>
              <li>Integrated hard resource depletion (battery capacity & charging points).</li>
              <li>Added `get_global_state()` method tailored for CTDE mixing networks.</li>
            </ul>
          </div>
        </div>
      </div>

    </div>
  `;
}
