/**
 * api.js — All network calls to the Demeter backend (port 5000).
 * Exports: checkHealth, fetchLatest, fetchHistory, fetchMetrics, fetchStatus, postPredict
 */

const API = window.location.origin + '/api';

export async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`);
    return r.ok;
  } catch { return false; }
}

export async function fetchLatest() {
  try {
    const r = await fetch(`${API}/latest`);
    if (r.status === 404) return null;   // no diagnosis yet — expected
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } catch (e) {
    console.warn('[api] fetchLatest:', e.message);
    return null;
  }
}

export async function fetchHistory(limit = 10) {
  try {
    const r = await fetch(`${API}/history?limit=${limit}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    return d.records || [];
  } catch (e) {
    console.warn('[api] fetchHistory:', e.message);
    return [];
  }
}

export async function fetchMetrics() {
  try {
    const r = await fetch(`${API}/metrics`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } catch (e) {
    console.warn('[api] fetchMetrics:', e.message);
    return null;
  }
}

export async function fetchStatus() {
  try {
    const r = await fetch(`${API}/status`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } catch (e) {
    console.warn('[api] fetchStatus:', e.message);
    return null;
  }
}

/**
 * POST /api/predict — multipart form with image + sensor values.
 * @param {File} imageFile
 * @param {Object} sensors  { temperature, soil_moisture, sunlight_hours, humidity }
 * @returns {Object|null}
 */
export async function postPredict(imageFile, sensors) {
  const form = new FormData();
  form.append('image', imageFile);
  for (const [k, v] of Object.entries(sensors)) form.append(k, v);
  try {
    const r = await fetch(`${API}/predict`, { method: 'POST', body: form });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
    return d;
  } catch (e) {
    console.error('[api] postPredict:', e.message);
    throw e;
  }
}
