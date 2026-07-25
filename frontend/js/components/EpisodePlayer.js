/**
 * EpisodePlayer.js — Reusable EpisodePlayer component
 * Full implementation: Step 6 (Frontend).
 * [V2-READY]: Exposes clean API for V2 live inference integration.
 */
export class EpisodePlayer {
  constructor(container, options = {}) {
    this.container = container;
    this.options = options;
  }
  render() { throw new Error('EpisodePlayer.render() — implemented in Step 6'); }
  destroy() { this.container.innerHTML = ''; }
}
