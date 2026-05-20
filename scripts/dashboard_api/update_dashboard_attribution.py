import sys

with open('src/frontend/dashboard.html', 'r', encoding='utf-8') as f:
    text = f.read()

replacements = {
    '<div class="card-title"><span class="card-title-icon">🔬</span>CNN Disease Detection</div>':
    '<div class="card-title"><span class="card-title-icon">🔬</span>CNN Disease Detection</div>\n          <div style="font-size:0.75rem; color:var(--muted); margin-bottom:8px;">Model: PlantVillage CNN</div>',
    
    '<div class="card-title"><span class="card-title-icon">⚖️</span>Biomass Estimation</div>\n          <div style="font-size:1.6rem; font-weight:700; color:var(--text); margin-top:20px;" id="biomass-pred">—</div>\n          <div style="font-size:0.8rem; color:var(--muted); margin-bottom:10px;">CNN Fresh Weight (g)</div>':
    '<div class="card-title"><span class="card-title-icon">⚖️</span>Biomass Estimation</div>\n          <div style="font-size:0.75rem; color:var(--muted); margin-bottom:8px;">Model: Biomass CNN (Dataset 6)</div>\n          <div style="font-size:1.6rem; font-weight:700; color:var(--text); margin-top:20px;" id="biomass-pred">—</div>',
    
    '<div class="card-title"><span class="card-title-icon">🌾</span>Tiller Count</div>\n          <div style="font-size:1.6rem; font-weight:700; color:var(--text); margin-top:20px;" id="tiller-pred">—</div>\n          <div style="font-size:0.8rem; color:var(--muted); margin-bottom:10px;">CNN Tiller Count</div>':
    '<div class="card-title"><span class="card-title-icon">🌾</span>Tiller Count</div>\n          <div style="font-size:0.75rem; color:var(--muted); margin-bottom:8px;">Model: Tiller CNN (Dataset 3)</div>\n          <div style="font-size:1.6rem; font-weight:700; color:var(--text); margin-top:20px;" id="tiller-pred">—</div>',
    
    '<div class="card-title"><span class="card-title-icon">💧</span>Water Stress (Bellwether)</div>\n          <div style="font-size:1.6rem; font-weight:700; color:var(--text); margin-top:20px;" id="bellwether-pred">—</div>\n          <div style="font-size:0.8rem; color:var(--muted); margin-bottom:10px;">CNN Vision Status</div>':
    '<div class="card-title"><span class="card-title-icon">💧</span>Water Stress</div>\n          <div style="font-size:0.75rem; color:var(--muted); margin-bottom:8px;">Model: Bellwether Multimodal (CNN+RF)</div>\n          <div style="font-size:1.6rem; font-weight:700; color:var(--text); margin-top:20px;" id="bellwether-pred">—</div>',
    
    '<div class="card-title"><span class="card-title-icon">💚</span>Health Status</div>':
    '<div class="card-title"><span class="card-title-icon">💚</span>Health Status</div>\n          <div style="font-size:0.7rem; color:var(--amber); margin-bottom:8px; line-height:1.2;">⚠️ Note: Health Score and Overall Status are currently calculated using arbitrary heuristic thresholds, not real data or facts.</div>',
    
    '<div class="card-title"><span class="card-title-icon">⚠️</span>Stress Diagnosis</div>':
    '<div class="card-title"><span class="card-title-icon">⚠️</span>Stress Diagnosis</div>\n          <div style="font-size:0.75rem; color:var(--amber); margin-bottom:8px;">⚠️ Heuristic Rule-Based Engine</div>',
    
    '<div class="card-title"><span class="card-title-icon">📈</span>Growth Trajectory</div>':
    '<div class="card-title"><span class="card-title-icon">📈</span>Growth Trajectory</div>\n          <div style="font-size:0.75rem; color:var(--muted); margin-bottom:8px;">Model: Danforth RF Regressor</div>',
    
    '<div class="card-title"><span class="card-title-icon">⚙️</span>Command Center</div>':
    '<div class="card-title"><span class="card-title-icon">⚙️</span>Command Center</div>\n          <div style="font-size:0.75rem; color:var(--amber); margin-bottom:8px;">⚠️ Heuristic Rule-Based Engine</div>'
}

for k, v in replacements.items():
    if k in text:
        text = text.replace(k, v)
    else:
        print(f"Warning: could not find replacement target for {k[:30]}...")

with open('src/frontend/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(text)

print("dashboard.html updated successfully!")
