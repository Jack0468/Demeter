/**
 * upload.js — Image drop-zone, file picker, and sensor sliders.
 * Exports: initUpload()
 */
import { postPredict } from './api.js';

let selectedFile = null;

export function initUpload(onResult) {
  const zone      = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const analyzeBtn= document.getElementById('analyze-btn');

  if (!zone) return;

  // ── Drag & drop ──────────────────────────────────────────────
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer?.files?.[0];
    if (file && file.type.startsWith('image/')) setFile(file);
  });
  zone.addEventListener('click', () => fileInput?.click());

  // ── File picker ───────────────────────────────────────────────
  if (fileInput) {
    fileInput.addEventListener('change', e => {
      const file = e.target.files?.[0];
      if (file) setFile(file);
    });
  }

  // ── Sliders ───────────────────────────────────────────────────
  const sliders = ['temperature', 'soil_moisture', 'sunlight_hours', 'humidity'];
  sliders.forEach(name => {
    const slider   = document.getElementById(`slider-${name}`);
    const valueEl  = document.getElementById(`slider-val-${name}`);
    const toggleEl = document.getElementById(`toggle-${name}`);
    if (!slider) return;

    const update = () => {
      const on  = !toggleEl || toggleEl.checked;
      const val = on ? slider.value : '--';
      if (valueEl) valueEl.textContent = val;
      slider.disabled = !on;
      slider.style.opacity = on ? '1' : '0.3';
    };

    slider.addEventListener('input', update);
    if (toggleEl) toggleEl.addEventListener('change', update);
    update();
  });

  // ── Analyze button ────────────────────────────────────────────
  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', async () => {
      if (!selectedFile) { alert('Please select a plant image first.'); return; }
      analyzeBtn.disabled = true;
      analyzeBtn.textContent = 'Processing inference…';
      
      const grid = document.getElementById('live-grid');
      if (grid) grid.style.opacity = '0.5';

      const sensors = {};
      ['temperature', 'soil_moisture', 'sunlight_hours', 'humidity'].forEach(name => {
        const toggle = document.getElementById(`toggle-${name}`);
        const slider = document.getElementById(`slider-${name}`);
        const on = !toggle || toggle.checked;
        sensors[name] = on ? parseFloat(slider?.value ?? 0) : (
          name === 'temperature' ? 25 : name === 'soil_moisture' ? 50 : name === 'sunlight_hours' ? 6 : 50
        );
      });

      try {
        const result = await postPredict(selectedFile, sensors);
        if (typeof onResult === 'function') onResult(result);
      } catch (e) {
        alert('Analysis failed: ' + e.message);
      } finally {
        analyzeBtn.textContent = 'Analyse Plant';
        analyzeBtn.disabled = false;
        const grid = document.getElementById('live-grid');
        if (grid) grid.style.opacity = '1';
      }
    });
  }
}

function setFile(file) {
  selectedFile = file;
  const zone = document.getElementById('drop-zone');
  const text = document.getElementById('drop-zone-text');
  if (!zone) return;
  // show preview
  const reader = new FileReader();
  reader.onload = e => {
    let img = zone.querySelector('img');
    if (!img) { img = document.createElement('img'); zone.appendChild(img); }
    img.src = e.target.result;
    img.style.display = 'block';
    if (text) text.style.display = 'none';
  };
  reader.readAsDataURL(file);
  const btn = document.getElementById('analyze-btn');
  if (btn) {
    btn.style.display = 'inline-block';
    btn.textContent = 'Analyse Plant';
  }
}
