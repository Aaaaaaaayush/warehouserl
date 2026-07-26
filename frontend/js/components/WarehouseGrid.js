/**
 * WarehouseGrid.js — HTML5 Canvas rendering engine for warehouse environment
 *
 * WHY THIS EXISTS:
 *   Renders the warehouse grid, cell types, and agent dots at 60 FPS on HTML5 Canvas.
 *   
 * [V2-READY]: Exposes a clean updateFrame(frame) method that V2's WebSocket
 *   live streaming client calls directly for real-time inference rendering.
 */

const CELL_TYPES = {
  EMPTY: 0,
  SHELF: 1,
  DISPATCH: 2,
  CHARGER: 3,
  WALL: 4,
  AGENT: 5,
  ITEM: 6,
};

const COLORS = {
  bg: '#04070f',
  empty: '#0a1628',
  shelf: '#1a2d4a',
  item: '#ffaa00',
  dispatch: '#0066ff',
  charger: '#00cc88',
  wall: '#2a3a4a',
  agentMoving: '#4a6080',
  agentCarrying: '#0066ff',
  agentLowBat: '#ffaa00',
  agentFrozen: '#ff3355',
  gridLine: '#14233c',
};

export class WarehouseGrid {
  /**
   * @param {HTMLElement} container  Parent container element
   * @param {object} gridConfig       Grid parameters {width, height, shelves, dispatch_points, charging_stations, obstacles}
   */
  constructor(container, gridConfig) {
    this.container = container;
    this.cfg = gridConfig;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.cellPx = 32;
    
    this.container.innerHTML = '';
    this.container.appendChild(this.canvas);
    
    this.resize();
    this.drawBaseGrid();
  }

  resize() {
    this.cellPx = Math.floor(Math.min(
      (this.container.clientWidth || 500) / this.cfg.width,
      (this.container.clientHeight || 500) / this.cfg.height,
      40
    ));
    if (this.cellPx < 16) this.cellPx = 16;

    this.canvas.width = this.cfg.width * this.cellPx;
    this.canvas.height = this.cfg.height * this.cellPx;
  }

  /**
   * Render base grid background, shelves, chargers, dispatch points, walls.
   */
  drawBaseGrid(itemsOnShelves = []) {
    const ctx = this.ctx;
    const px = this.cellPx;
    const W = this.cfg.width;
    const H = this.cfg.height;

    ctx.fillStyle = COLORS.empty;
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw borders & walls
    ctx.fillStyle = COLORS.wall;
    ctx.fillRect(0, 0, W * px, px);
    ctx.fillRect(0, (H - 1) * px, W * px, px);
    ctx.fillRect(0, 0, px, H * px);
    ctx.fillRect((W - 1) * px, 0, px, H * px);

    if (this.cfg.obstacles) {
      for (const [r, c] of this.cfg.obstacles) {
        ctx.fillRect(c * px, r * px, px, px);
      }
    }

    // Draw Shelves
    ctx.fillStyle = COLORS.shelf;
    for (const [r, c] of this.cfg.shelves) {
      ctx.fillRect(c * px + 2, r * px + 2, px - 4, px - 4);
    }

    // Draw Items on shelves (amber dots)
    ctx.fillStyle = COLORS.item;
    for (const [r, c] of itemsOnShelves) {
      ctx.beginPath();
      ctx.arc(c * px + px / 2, r * px + px / 2, px / 5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw Chargers
    ctx.fillStyle = COLORS.charger;
    for (const [r, c] of this.cfg.charging_stations) {
      ctx.fillRect(c * px + 3, r * px + 3, px - 6, px - 6);
    }

    // Draw Dispatch Points
    ctx.fillStyle = COLORS.dispatch;
    for (const [r, c] of this.cfg.dispatch_points) {
      ctx.fillRect(c * px + 3, r * px + 3, px - 6, px - 6);
    }

    // Grid lines
    ctx.strokeStyle = COLORS.gridLine;
    ctx.lineWidth = 1;
    for (let c = 0; c <= W; c++) {
      ctx.beginPath();
      ctx.moveTo(c * px, 0);
      ctx.lineTo(c * px, H * px);
      ctx.stroke();
    }
    for (let r = 0; r <= H; r++) {
      ctx.beginPath();
      ctx.moveTo(0, r * px);
      ctx.lineTo(W * px, r * px);
      ctx.stroke();
    }
  }

  /**
   * [V2-READY] Update grid with a new step frame data.
   * @param {object} frame  {step, agents, items_on_shelves}
   */
  updateFrame(frame) {
    this.drawBaseGrid(frame.items_on_shelves || []);

    const ctx = this.ctx;
    const px = this.cellPx;

    // Draw Agents
    for (const a of frame.agents) {
      if (a.frozen || a.row < 0) continue;

      const cx = a.col * px + px / 2;
      const cy = a.row * px + px / 2;
      const radius = px / 2 - 3;

      let color = COLORS.agentMoving;
      if (a.frozen) color = COLORS.agentFrozen;
      else if (a.battery < 20) color = COLORS.agentLowBat;
      else if (a.carrying) color = COLORS.agentCarrying;

      // Agent body
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(cx, cy, Math.max(radius, 4), 0, Math.PI * 2);
      ctx.fill();

      // Carrying item indicator dot
      if (a.carrying) {
        ctx.fillStyle = COLORS.item;
        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
}
