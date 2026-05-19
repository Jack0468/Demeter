/**
 * live.js — Live Diagnostics tab orchestration.
 * Exports: initLiveTab(), refreshLiveTab(diagnosis, history)
 */
import {
  renderDiseaseCard,
  renderHealthCard,
  renderStressCards,
  renderTrajectory,
  renderCommandAndRecs,
  renderSensorReadouts,
  renderHistoryTable,
  renderLastUpdated,
  renderMultiModels,
} from './render.js';

export function initLiveTab() {
  // Initialize health ring SVG dimensions
  const fill = document.getElementById('health-ring-fill');
  if (fill) {
    const r = 54;
    const circ = 2 * Math.PI * r;
    fill.style.strokeDasharray  = circ;
    fill.style.strokeDashoffset = circ;
  }
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
  renderHealthCard(diagnosis);
  renderStressCards(diagnosis.stress_diagnosis);
  renderTrajectory(diagnosis.trajectory_7day);
  renderCommandAndRecs(diagnosis);
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
