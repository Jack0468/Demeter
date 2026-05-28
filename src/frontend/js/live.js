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
    hideEmptyState();
    renderDiseaseCard(null);
    renderHybridDiseaseCard(null);
    renderHealthCard(null);
    renderSensorReadouts(null);
    renderHistoryTable(null);
    renderLastUpdated(null);
    renderMultiModels(null);
    return;
  }
  hideEmptyState();

  renderDiseaseCard(diagnosis.cnn_result);
  const svm_pred = diagnosis.hierarchical_svm_prediction || diagnosis.hybrid_prediction;
  renderHybridDiseaseCard(svm_pred);
  renderHealthCard(diagnosis);
  renderSensorReadouts(diagnosis.sensors);
  renderHistoryTable(history);
  renderLastUpdated(diagnosis.timestamp);
  renderMultiModels(diagnosis);
}

function showEmptyState() {
  const grid = document.getElementById('live-grid');
  if (grid) grid.style.opacity = '1';
}

function hideEmptyState() {
  const grid = document.getElementById('live-grid');
  if (grid) grid.style.opacity = '1';
}
