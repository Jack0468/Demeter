/**
 * performance.js — Model Performance tab.
 * Exports: initPerformanceTab(), refreshPerformanceTab(metrics)
 */
import { renderMetrics } from './render.js';

export function initPerformanceTab() {
  // nothing to set up statically — data is fetched on tab activation
}

/**
 * Update model performance panels.
 * @param {Object|null} metrics — response from /api/metrics
 */
export function refreshPerformanceTab(metrics) {
  if (!metrics) {
    const el = document.getElementById('perf-empty');
    if (el) el.style.display = 'flex';
    return;
  }
  const el = document.getElementById('perf-empty');
  if (el) el.style.display = 'none';
  renderMetrics(metrics);
}
