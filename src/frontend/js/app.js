/**
 * app.js — Main entry point. Wires tabs, refresh loop, upload handler.
 */
import { checkHealth, fetchLatest, fetchHistory, fetchMetrics, fetchStatus } from './api.js';
import { renderApiPill, renderModelPill } from './render.js';
import { initLiveTab, refreshLiveTab } from './live.js';
import { initPerformanceTab, refreshPerformanceTab } from './performance.js';
import { initUpload } from './upload.js';

const REFRESH_MS = 5000;
let activeTab = 'live';
let metricsLoaded = false;

// ── Tab switching ──────────────────────────────────────────────
function switchTab(name) {
  activeTab = name;
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === name);
  });
  document.querySelectorAll('.tab-pane').forEach(pane => {
    pane.classList.toggle('active', pane.id === `tab-${name}`);
  });
  if (name === 'performance' && !metricsLoaded) loadMetrics();
}

async function loadMetrics() {
  const metrics = await fetchMetrics();
  refreshPerformanceTab(metrics);
  metricsLoaded = true;
}

// ── Main refresh ───────────────────────────────────────────────
async function refresh() {
  const online = await checkHealth();
  renderApiPill(online);

  if (!online) return;

  const [diagnosis, history] = await Promise.all([
    fetchLatest(),
    fetchHistory(10),
  ]);

  refreshLiveTab(diagnosis, history);

  if (activeTab === 'performance' && !metricsLoaded) loadMetrics();
}

// ── Init ───────────────────────────────────────────────────────
async function init() {
  // Tab buttons
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });
  switchTab('live');

  initLiveTab();
  initPerformanceTab();

  // Upload module: when predict returns, refresh live tab
  initUpload(async (result) => {
    // result is the predict response — treat as latest diagnosis
    refreshLiveTab(result, await fetchHistory(10));
  });

  // Fetch model pill once
  const status = await fetchStatus();
  renderModelPill(status);

  // Initial data load + recurring refresh
  await refresh();
  setInterval(refresh, REFRESH_MS);
}

document.addEventListener('DOMContentLoaded', init);
