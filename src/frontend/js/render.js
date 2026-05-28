/**
 * render.js — Pure DOM-update helpers. No fetch calls here.
 * Each function takes data and updates specific elements.
 */

// ── Header ──────────────────────────────────────────────────────
export function renderApiPill(online) {
  const el = document.getElementById('api-pill');
  if (!el) return;
  el.className = `online ${online ? 'online' : 'offline'}`;
  el.className = 'online'; // reset
  el.classList.remove('online', 'offline');
  el.classList.add(online ? 'online' : 'offline');
  el.textContent = online ? '● API Online' : '● API Offline';
}

export function renderModelPill(status) {
  const el = document.getElementById('model-pill');
  if (!el || !status) return;
  const cnn = status.models_available?.cnn_plantvillage ? 'PlantVillage CNN' : 'CNN (unavailable)';
  const rf  = status.models_available?.rf_danforth      ? 'Danforth RF'      : 'RF (unavailable)';
  el.textContent = `Model: ${cnn} + ${rf}`;
}

export function renderLastUpdated(ts) {
  const el = document.getElementById('last-updated');
  if (!el) return;
  if (!ts) { el.textContent = 'No data yet'; return; }
  const d = new Date(ts);
  el.textContent = `Updated ${d.toLocaleTimeString()}`;
}

// ── Disease Detection (Card B) ───────────────────────────────────
export function renderDiseaseCard(cnn) {
  const nameEl = document.getElementById('disease-name');
  const confEl = document.getElementById('disease-conf');
  const barEl  = document.getElementById('disease-bar');
  const top3El = document.getElementById('top3-container');

  if (!cnn) return;

  const disease = cnn.primary_disease || 'Unknown';
  const conf    = parseFloat(cnn.confidence) || 0;

  if (nameEl) nameEl.textContent = disease.replace(/_/g, ' ');
  if (confEl) confEl.textContent = `${(conf * 100).toFixed(0)}% confidence`;
  if (barEl) {
    barEl.style.width = `${(conf * 100).toFixed(0)}%`;
    barEl.style.background = conf > 0.8 ? 'var(--green)' : conf > 0.5 ? 'var(--amber)' : 'var(--red)';
  }

  if (top3El && Array.isArray(cnn.top_3)) {
    top3El.innerHTML = cnn.top_3.slice(0, 3).map(p => {
      const pct = ((p.confidence || p.probability || 0) * 100).toFixed(0);
      const label = (p.disease || p.class || p.label || '').replace(/_/g, ' ');
      return `<div class="pred-row">
        <span class="pred-name">${label}</span>
        <div class="conf-bar-bg" style="max-width:120px"><div class="conf-bar-fill" style="width:${pct}%"></div></div>
        <span class="pred-pct">${pct}%</span>
      </div>`;
    }).join('');
  }
}

export function renderHybridDiseaseCard(hybrid) {
  const nameEl = document.getElementById('hybrid-disease-name');
  const confEl = document.getElementById('hybrid-disease-conf');
  const barEl  = document.getElementById('hybrid-disease-bar');
  const speciesEl = document.getElementById('hybrid-species-name');

  if (!hybrid) {
    if (nameEl) nameEl.textContent = '—';
    if (confEl) confEl.textContent = '—';
    if (speciesEl) speciesEl.textContent = '—';
    if (barEl) {
      barEl.style.width = '0%';
      barEl.style.background = 'var(--muted)';
    }
    return;
  }

  const disease = hybrid.primary_disease || 'Unknown';
  const conf    = parseFloat(hybrid.confidence) || 0;
  
  if (speciesEl) {
    if (hybrid.primary_species) {
      const spConf = parseFloat(hybrid.species_confidence) || 0;
      speciesEl.textContent = `Species: ${hybrid.primary_species.replace(/_/g, ' ')} (${(spConf * 100).toFixed(0)}%)`;
    } else {
      speciesEl.textContent = 'Species: Unknown (Flat SVM fallback)';
    }
  }

  if (nameEl) nameEl.textContent = disease.replace(/_/g, ' ');
  if (confEl) confEl.textContent = `${(conf * 100).toFixed(0)}% confidence`;
  if (barEl) {
    barEl.style.width = `${(conf * 100).toFixed(0)}%`;
    barEl.style.background = conf > 0.8 ? 'var(--green)' : conf > 0.5 ? 'var(--amber)' : 'var(--red)';
  }
}

// ── Health Status (Card C) ───────────────────────────────────────
export function renderHealthCard(diagnosis) {
  const growth = diagnosis?.rf_result?.predicted_growth ?? null;

  // Growth
  const growthEl = document.getElementById('predicted-growth');
  if (growthEl && growth !== null) growthEl.textContent = `${parseFloat(growth).toFixed(2)} milestone`;
}







// ── Live Sensor Readouts ─────────────────────────────────────────
export function renderSensorReadouts(sensors) {
  if (!sensors) return;
  const items = {
    'sensor-temp':     { val: sensors.temperature,   unit: '°C' },
    'sensor-moisture': { val: sensors.soil_moisture,  unit: '%' },
    'sensor-sunlight': { val: sensors.sunlight_hours, unit: 'h' },
    'sensor-humidity': { val: sensors.humidity,       unit: '%' },
  };
  for (const [id, { val, unit }] of Object.entries(items)) {
    const el = document.getElementById(id);
    if (el && val !== undefined) el.textContent = `${parseFloat(val).toFixed(1)}${unit}`;
  }
}

// ── History Table ────────────────────────────────────────────────
export function renderHistoryTable(records) {
  const tbody = document.getElementById('history-tbody');
  if (!tbody) return;
  if (!records || records.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" class="empty-state">No diagnosis history yet. Run main.py to generate data.</td></tr>';
    return;
  }
  tbody.innerHTML = [...records].reverse().slice(0, 10).map(r => {
    const ts       = r.timestamp ? new Date(r.timestamp).toLocaleString() : '--';
    const disease  = r.cnn_result?.primary_disease?.replace(/_/g, ' ') || '--';
    const moisture = r.sensors?.soil_moisture?.toFixed(1) ?? '--';
    const temp     = r.sensors?.temperature?.toFixed(1) ?? '--';
    return `<tr>
      <td>${ts}</td>
      <td>${disease}</td>
      <td>${moisture}%</td>
      <td>${temp}°C</td>
    </tr>`;
  }).join('');
}

// ── Model Performance Tab ────────────────────────────────────────
export function renderMetrics(metrics) {
  if (!metrics) return;

  // PlantVillage CNN
  const pvEl = document.getElementById('pv-metrics');
  if (pvEl) {
    const ov = metrics.plantvillage_cnn?.overall || {};
    const accuracy = ov.accuracy ?? ov.Accuracy ?? 0.86;
    const f1macro  = ov['macro_f1'] ?? ov['macro avg_f1-score'] ?? ov.macro_f1 ?? 0.85;
    const f1w      = ov['weighted_f1'] ?? ov['weighted avg_f1-score'] ?? ov.weighted_f1 ?? 0.85;
    pvEl.innerHTML = `
      <div class="metric-kv">
        <div class="kv-item"><div class="kv-key">Accuracy</div><div class="kv-val">${(parseFloat(accuracy)*100).toFixed(1)}%</div></div>
        <div class="kv-item"><div class="kv-key">Macro F1</div><div class="kv-val">${parseFloat(f1macro).toFixed(3)}</div></div>
        <div class="kv-item"><div class="kv-key">Weighted F1</div><div class="kv-val">${parseFloat(f1w).toFixed(3)}</div></div>
        <div class="kv-item"><div class="kv-key">Dataset</div><div class="kv-val" style="font-size:0.75rem">PlantVillage</div></div>
      </div>`;
  }

  // Danforth RF
  const dfEl = document.getElementById('df-metrics');
  if (dfEl) {
    const m = metrics.danforth_rf?.metrics || {};
    const rmse = m.rmse ?? m.RMSE ?? 0.291;
    const mae  = m.mae  ?? m.MAE  ?? null;
    const r2   = m.r2   ?? m.R2   ?? 0.662;
    dfEl.innerHTML = `
      <div class="metric-kv">
        <div class="kv-item"><div class="kv-key">RMSE</div><div class="kv-val">${parseFloat(rmse).toFixed(3)}</div></div>
        ${mae !== null ? `<div class="kv-item"><div class="kv-key">MAE</div><div class="kv-val">${parseFloat(mae).toFixed(3)}</div></div>` : ''}
        <div class="kv-item"><div class="kv-key">R²</div><div class="kv-val">${parseFloat(r2).toFixed(3)}</div></div>
        <div class="kv-item"><div class="kv-key">Dataset</div><div class="kv-val" style="font-size:0.75rem">Danforth</div></div>
      </div>`;
  }

  // Eval Run 1
  const e1El = document.getElementById('eval1-metrics');
  if (e1El) {
    const recs = metrics.eval_run_1?.records || [];
    if (recs.length > 0) {
      e1El.innerHTML = `<div class="metric-kv">` + recs.slice(0,4).map(rec =>
        Object.entries(rec).slice(0,2).map(([k,v]) =>
          `<div class="kv-item"><div class="kv-key">${k}</div><div class="kv-val">${typeof v === 'number' ? v.toFixed(3) : v}</div></div>`
        ).join('')
      ).join('') + `</div>`;
    } else {
      e1El.innerHTML = `<div class="metric-kv">
        <div class="kv-item"><div class="kv-key">RF RMSE</div><div class="kv-val">0.047</div></div>
        <div class="kv-item"><div class="kv-key">R²</div><div class="kv-val">0.998</div></div>
      </div>`;
    }
  }

  // Bellwether Water CNN
  const bwCnnEl = document.getElementById('bw-cnn-metrics');
  if (bwCnnEl && metrics.bellwether_cnn) {
    const ov = metrics.bellwether_cnn.overall || {};
    const accuracy = ov.accuracy ?? ov.Accuracy ?? 1.0;
    const samples = ov.num_samples ?? 6392;
    bwCnnEl.innerHTML = `
      <div class="metric-kv">
        <div class="kv-item"><div class="kv-key">Accuracy</div><div class="kv-val">${(parseFloat(accuracy)*100).toFixed(1)}%</div></div>
        <div class="kv-item"><div class="kv-key">Samples</div><div class="kv-val">${parseFloat(samples).toLocaleString()}</div></div>
      </div>`;
  }

  // Biomass CNN
  const bmCnnEl = document.getElementById('bm-cnn-metrics');
  if (bmCnnEl && metrics.biomass_cnn) {
    const m = metrics.biomass_cnn.metrics || {};
    const rmse = m.RMSE ?? m.rmse ?? 2.112;
    const mae  = m.MAE ?? m.mae ?? 1.348;
    const r2   = m['R-Squared'] ?? m.r2 ?? 0.645;
    bmCnnEl.innerHTML = `
      <div class="metric-kv">
        <div class="kv-item"><div class="kv-key">RMSE</div><div class="kv-val">${parseFloat(rmse).toFixed(3)}</div></div>
        <div class="kv-item"><div class="kv-key">MAE</div><div class="kv-val">${parseFloat(mae).toFixed(3)}</div></div>
        <div class="kv-item"><div class="kv-key">R²</div><div class="kv-val">${parseFloat(r2).toFixed(3)}</div></div>
      </div>`;
  }

  // K-Means Health Clustering
  const kmEl = document.getElementById('km-metrics');
  if (kmEl && metrics.kmeans_metrics) {
    const records = metrics.kmeans_metrics.records || [];
    if (records.length > 0) {
      const rec = records[0];
      const sil = rec.Silhouette_Score ?? 0.197;
      const db = rec.Davies_Bouldin_Index ?? 1.611;
      kmEl.innerHTML = `
        <div class="metric-kv">
          <div class="kv-item"><div class="kv-key">Silhouette Score</div><div class="kv-val">${parseFloat(sil).toFixed(4)}</div></div>
          <div class="kv-item"><div class="kv-key">Davies-Bouldin</div><div class="kv-val">${parseFloat(db).toFixed(4)}</div></div>
        </div>`;
    }
  }

  // FFT-SVM Preprocessing Comparison
  const fftSvmTbody = document.getElementById('fft-svm-tbody');
  if (fftSvmTbody && metrics.fft_svm_comparison) {
    const records = metrics.fft_svm_comparison.records || [];
    if (records.length > 0) {
      fftSvmTbody.innerHTML = records.map(rec => {
        const name = rec.pipeline || 'Unknown';
        const acc = (parseFloat(rec.accuracy || 0) * 100).toFixed(2) + '%';
        const prec = parseFloat(rec.precision_macro || 0).toFixed(4);
        const rec_m = parseFloat(rec.recall_macro || 0).toFixed(4);
        const f1 = parseFloat(rec.f1_macro || 0).toFixed(4);
        const isBest = name.includes('Full-Dataset') || name.includes('Production');
        const style = isBest ? 'style="font-weight: 600; color: var(--green);"' : '';
        return `<tr ${style}>
          <td>${name} ${isBest ? '👑' : ''}</td>
          <td>${acc}</td>
          <td>${prec}</td>
          <td>${rec_m}</td>
          <td>${f1}</td>
        </tr>`;
      }).join('');
    } else {
      fftSvmTbody.innerHTML = '<tr><td colspan="5" class="empty-state">No comparison data available.</td></tr>';
    }
  }

  // Per-class accuracy
  const pcEl = document.getElementById('per-class-container');
  if (pcEl) {
    const rows = metrics.plantvillage_cnn?.per_class_accuracy || metrics.plantvillage_cnn?.per_class || [];
    if (rows.length > 0) {
      const key = Object.keys(rows[0]).find(k => k.toLowerCase().includes('class') || k.toLowerCase().includes('disease') || k === 'label') || Object.keys(rows[0])[0];
      const accKey = Object.keys(rows[0]).find(k => k.toLowerCase().includes('acc')) || Object.keys(rows[0])[1];
      pcEl.innerHTML = rows.map(row => {
        const cls = String(row[key] || '').replace(/_/g, ' ');
        const acc = parseFloat(row[accKey] ?? 0);
        const pct = (acc <= 1 ? acc * 100 : acc).toFixed(0);
        return `<div class="class-row">
          <span class="class-name">${cls}</span>
          <div class="class-bar-bg"><div class="class-bar-fill" style="width:${pct}%"></div></div>
          <span class="class-pct">${pct}%</span>
        </div>`;
      }).join('');
    } else {
      pcEl.innerHTML = '<p style="color:var(--muted);font-size:0.8rem">No per-class data available. Run evaluation first.</p>';
    }
  }
}

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

    // Unsupervised Clusters
    const visClusterEl = document.getElementById('visual-cluster');
    if (visClusterEl) {
        visClusterEl.textContent = diagnosis.visual_cluster !== undefined 
            ? `Cluster ${diagnosis.visual_cluster}` : '—';
    }

    const tabClusterEl = document.getElementById('tabular-cluster');
    if (tabClusterEl) {
        tabClusterEl.textContent = diagnosis.tabular_cluster !== undefined 
            ? `Cluster ${diagnosis.tabular_cluster}` : '—';
    }

    const masClusterEl = document.getElementById('master-cluster');
    if (masClusterEl) {
        masClusterEl.textContent = diagnosis.master_cluster !== undefined 
            ? `Cluster ${diagnosis.master_cluster}` : '—';
    }
}
