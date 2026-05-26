/**
 * live.js — Live Diagnostics tab orchestration.
 * Exports: initLiveTab(), refreshLiveTab(diagnosis, history)
 */
import {
  renderDiseaseCard,
  renderHybridDiseaseCard,
  renderHealthCard,
  renderSensorReadouts,
  renderHistoryTable,
  renderLastUpdated,
  renderMultiModels,
} from './render.js';

export function initLiveTab() {
  // No initialization needed for simplified UI
}

/**
 * Update all Live Diagnostics panels from fresh API data.
 * @param {Object|null} diagnosis  — response from /api/latest
 * @param {Array}       history    — response from /api/history
 */
export function refreshLiveTab(diagnosis, history) {
  if (!diagnosis) {
    showEmptyState();
    return;
  }
  hideEmptyState();

  renderDiseaseCard(diagnosis.cnn_result);
  renderHybridDiseaseCard(diagnosis.hybrid_prediction);
  renderHealthCard(diagnosis);
  renderSensorReadouts(diagnosis.sensors);
  renderHistoryTable(history);
  renderLastUpdated(diagnosis.timestamp);
  renderMultiModels(diagnosis);
}

function showEmptyState() {
  const grid = document.getElementById('live-grid');
  if (grid) grid.style.opacity = '0.3';
}

function hideEmptyState() {
  const grid = document.getElementById('live-grid');
  if (grid) grid.style.opacity = '1';
}
