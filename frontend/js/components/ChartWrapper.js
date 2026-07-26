/**
 * ChartWrapper.js — Plotly.js wrapper matching design system tokens
 *
 * WHY THIS EXISTS:
 *   Wraps Plotly chart initialization with consistent dark theme colors,
 *   margins, font choices, and animation trace helpers.
 */

const DARK_LAYOUT = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: '#0a1628',
  font: {
    family: 'JetBrains Mono, monospace',
    color: '#e8edf5',
    size: 11,
  },
  margin: { l: 40, r: 20, t: 30, b: 35 },
  xaxis: {
    gridcolor: '#14233c',
    zerolinecolor: '#1a2d4a',
    tickfont: { color: '#4a6080', size: 10 },
  },
  yaxis: {
    gridcolor: '#14233c',
    zerolinecolor: '#1a2d4a',
    tickfont: { color: '#4a6080', size: 10 },
  },
};

export class ChartWrapper {
  /**
   * @param {HTMLElement} container  DOM element to attach Plotly chart
   * @param {string} title           Chart title
   * @param {string} lineColor       Line color hex/rgba
   */
  constructor(container, title, lineColor = '#0066ff') {
    this.container = container;
    this.title = title;
    this.lineColor = lineColor;
  }

  /**
   * Render or update time-series data.
   * @param {Array<number>} xValues  Episode numbers
   * @param {Array<number>} yValues  Metric values
   */
  render(xValues, yValues) {
    if (!window.Plotly) return;

    const trace = {
      x: xValues,
      y: yValues,
      type: 'scatter',
      mode: 'lines',
      line: { color: this.lineColor, width: 2 },
    };

    const layout = {
      ...DARK_LAYOUT,
      title: {
        text: this.title,
        font: { size: 12, color: '#e8edf5' },
        x: 0.02,
      },
    };

    window.Plotly.newPlot(this.container, [trace], layout, {
      responsive: true,
      displayModeBar: false,
    });
  }
}
