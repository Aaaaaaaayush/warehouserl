/**
 * router.js — Client-side panel router
 *
 * WHY THIS EXISTS:
 *   The app is a single HTML page. Clicking sidebar nav items
 *   should swap the content in #main-panel without a full page reload.
 *   This is called a Single-Page Application (SPA) pattern.
 *
 *   The router:
 *   1. Listens for nav clicks
 *   2. Dynamically imports the corresponding panel module
 *   3. Calls panel.render() which returns an HTML string or DOM node
 *   4. Injects it into #main-panel
 *   5. Updates the active nav state and topbar breadcrumb
 *
 * ADDING A NEW PANEL IN V2:
 *   1. Add one entry to _PANELS below
 *   2. Create js/panels/mypanel.js with an exported render() function
 *   3. Add one <li> to index.html sidebar
 *   That's it. Zero other changes required. [V2-READY]
 */

import { state } from './state.js';

// Panel registry — maps nav data-panel attr to module path
const _PANELS = {
  overview:     () => import('../panels/overview.js'),
  scenarios:    () => import('../panels/scenarios.js'),
  warehouse:    () => import('../panels/warehouse.js'),
  training:     () => import('../panels/training.js'),
  behaviors:    () => import('../panels/behaviors.js'),
  architecture: () => import('../panels/architecture.js'),
  research:     () => import('../panels/research.js'),
  v2preview:    () => import('../panels/v2preview.js'),
};

const _mainPanel    = document.getElementById('main-panel');
const _breadcrumb   = document.getElementById('topbar-breadcrumb');
const _navItems     = document.querySelectorAll('.nav-item');

async function navigateTo(panelName) {
  if (!_PANELS[panelName]) {
    console.warn(`Router: unknown panel "${panelName}"`);
    return;
  }

  // Show loading state
  _mainPanel.innerHTML = `
    <div id="panel-loading" class="panel-loading">
      <span class="loading-dot"></span>
      <span class="loading-dot"></span>
      <span class="loading-dot"></span>
    </div>`;

  try {
    const module = await _PANELS[panelName]();
    const content = await module.render();

    // content can be a string (HTML) or a DOM Node
    if (typeof content === 'string') {
      _mainPanel.innerHTML = content;
    } else {
      _mainPanel.innerHTML = '';
      _mainPanel.appendChild(content);
    }

    // Run panel's post-render hook if provided (charts, canvas, etc.)
    if (module.afterRender) {
      await module.afterRender();
    }
  } catch (err) {
    _mainPanel.innerHTML = `
      <div style="padding: var(--space-4); color: var(--color-danger);">
        <p style="font-family: var(--font-mono);">Failed to load panel: ${panelName}</p>
        <pre style="margin-top: var(--space-2); font-size: var(--text-xs); color: var(--color-muted);">${err.message}</pre>
      </div>`;
    console.error(`Router: panel "${panelName}" failed to render`, err);
  }

  // Update state, active nav highlight, and breadcrumb
  state.set('activePanel', panelName);

  _navItems.forEach(item => {
    item.classList.toggle('active', item.dataset.panel === panelName);
  });

  const label = document.getElementById(`nav-${panelName}`)?.textContent?.trim() ?? panelName;
  _breadcrumb.textContent = label.replace('🔒', '').trim();
}

// ── Event listeners ────────────────────────────────────────────────────

_navItems.forEach(item => {
  item.addEventListener('click', (e) => {
    e.preventDefault();
    if (item.classList.contains('locked')) return;   // V2 panels: do nothing
    navigateTo(item.dataset.panel);
  });
});

// ── Initial load ───────────────────────────────────────────────────────

navigateTo(state.get('activePanel'));
