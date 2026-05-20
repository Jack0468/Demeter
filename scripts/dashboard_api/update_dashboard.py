import sys

with open('src/frontend/dashboard.html', 'r', encoding='utf-8') as f:
    text = f.read()

target_empty = 'Upload a plant image and click <strong style="color:var(--text)">Analyze Plant</strong>, or run <code style="color:var(--green)">main.py</code>.'
repl_empty = 'Upload a plant image and click <strong style="color:var(--text)">Analyze Plant</strong>.'

text = text.replace(target_empty, repl_empty)

target_row2 = '      <!-- ── ROW 2 ──────────────────────────────────────────── -->'
repl_row2 = '''      <!-- ── ROW 2: MULTI-MODEL PREDICTIONS ─────────────────── -->
      <div class="grid-3" style="margin-bottom:20px;">
        <!-- Card: Biomass -->
        <div class="card">
          <div class="card-title"><span class="card-title-icon">⚖️</span>Biomass Estimation</div>
          <div style="font-size:1.6rem; font-weight:700; color:var(--text); margin-top:20px;" id="biomass-pred">—</div>
          <div style="font-size:0.8rem; color:var(--muted); margin-bottom:10px;">CNN Fresh Weight (g)</div>
        </div>
        <!-- Card: Tiller Count -->
        <div class="card">
          <div class="card-title"><span class="card-title-icon">🌾</span>Tiller Count</div>
          <div style="font-size:1.6rem; font-weight:700; color:var(--text); margin-top:20px;" id="tiller-pred">—</div>
          <div style="font-size:0.8rem; color:var(--muted); margin-bottom:10px;">CNN Tiller Count</div>
        </div>
        <!-- Card: Bellwether Water Stress -->
        <div class="card">
          <div class="card-title"><span class="card-title-icon">💧</span>Water Stress (Bellwether)</div>
          <div style="font-size:1.6rem; font-weight:700; color:var(--text); margin-top:20px;" id="bellwether-pred">—</div>
          <div style="font-size:0.8rem; color:var(--muted); margin-bottom:10px;">CNN Vision Status</div>
        </div>
      </div>

      <!-- ── ROW 3 ──────────────────────────────────────────── -->'''

if target_row2 in text:
    text = text.replace(target_row2, repl_row2)
    # also rename ROW 3 to ROW 4
    text = text.replace('      <!-- ── ROW 3 ──────────────────────────────────────────── -->', '      <!-- ── ROW 4 ──────────────────────────────────────────── -->', 1)

with open('src/frontend/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(text)
print("Dashboard updated successfully!")
