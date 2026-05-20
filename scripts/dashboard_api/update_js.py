import sys

with open('src/frontend/js/render.js', 'r', encoding='utf-8') as f:
    render_text = f.read()

# Add renderMultiModels to render.js
new_func = """
// ── Multi-Model Predictions ──────────────────────────────────────
export function renderMultiModels(diagnosis) {
  if (!diagnosis) return;
  
  // Biomass
  const biomassEl = document.getElementById('biomass-pred');
  if (biomassEl) {
    biomassEl.textContent = diagnosis.biomass_prediction !== undefined 
      ? parseFloat(diagnosis.biomass_prediction).toFixed(2) 
      : '—';
  }
  
  // Tiller
  const tillerEl = document.getElementById('tiller-pred');
  if (tillerEl) {
    tillerEl.textContent = diagnosis.tiller_prediction !== undefined 
      ? Math.round(diagnosis.tiller_prediction)
      : '—';
  }
  
  // Bellwether
  const bellwetherEl = document.getElementById('bellwether-pred');
  if (bellwetherEl) {
    if (diagnosis.bellwether_water_stress && diagnosis.bellwether_water_stress.Vision_Status) {
      const status = diagnosis.bellwether_water_stress.Vision_Status.replace(/_/g, ' ');
      bellwetherEl.textContent = status;
      if (status.includes('Stressed')) {
        bellwetherEl.style.color = 'var(--red)';
      } else {
        bellwetherEl.style.color = 'var(--green)';
      }
    } else {
      bellwetherEl.textContent = '—';
      bellwetherEl.style.color = 'var(--text)';
    }
  }
}
"""

if "renderMultiModels" not in render_text:
    render_text += new_func

with open('src/frontend/js/render.js', 'w', encoding='utf-8') as f:
    f.write(render_text)

# Update live.js to import and use renderMultiModels
with open('src/frontend/js/live.js', 'r', encoding='utf-8') as f:
    live_text = f.read()

if "renderMultiModels" not in live_text:
    live_text = live_text.replace(
        "renderLastUpdated,",
        "renderLastUpdated,\n  renderMultiModels,"
    )
    live_text = live_text.replace(
        "renderLastUpdated(diagnosis.timestamp);",
        "renderLastUpdated(diagnosis.timestamp);\n  renderMultiModels(diagnosis);"
    )

with open('src/frontend/js/live.js', 'w', encoding='utf-8') as f:
    f.write(live_text)

print("JS files updated successfully!")
