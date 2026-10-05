/* =====================================================
   app.js — Smart Traffic Dashboard Frontend Logic
   =====================================================
   WebSocket → state_update events → DOM updates + charts
   All data comes from the FastAPI backend (server.py)
   which wraps the existing Python modules.
===================================================== */

const API_BASE = `${window.location.protocol}//${window.location.hostname}:8000`;
const WS_URL   = `ws://${window.location.hostname}:8000/ws/live`;

// ─────────────────────────────────────────────────────────
// GLOBAL STATE
// ─────────────────────────────────────────────────────────
let ws          = null;
let wsRetry     = null;
let lastState   = null;
let logBuffer   = [];

// Chart instances
let chartCount   = null;
let chartDensity = null;
let chartDonut   = null;
let chartLevel   = null;
let chartFlow    = null;
let chartHistory = null;
let chartReward  = null;
let chartPhase   = null;

// ─────────────────────────────────────────────────────────
// NAVIGATION
// ─────────────────────────────────────────────────────────
function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const page = document.getElementById(`page-${name}`);
  const nav  = document.querySelector(`.nav-item[data-page="${name}"]`);
  if (page) page.classList.add('active');
  if (nav)  nav.classList.add('active');
}

// ─────────────────────────────────────────────────────────
// WEBSOCKET
// ─────────────────────────────────────────────────────────
function connectWS() {
  if (ws && ws.readyState === WebSocket.OPEN) return;

  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    setWsStatus(true);
    addLog('INFO', 'WebSocket connected to backend.');
    if (wsRetry) { clearInterval(wsRetry); wsRetry = null; }
    document.getElementById('ws-url-display').textContent = WS_URL;
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'state_update') {
        lastState = data;
        updateDashboard(data);
      }
    } catch (e) {
      console.error('WS parse error:', e);
    }
  };

  ws.onclose = () => {
    setWsStatus(false);
    addLog('WARN', 'WebSocket disconnected. Retrying in 3s…');
    if (!wsRetry) {
      wsRetry = setInterval(connectWS, 3000);
    }
  };

  ws.onerror = (err) => {
    addLog('ERROR', 'WebSocket error. Is the backend running?');
  };
}

function setWsStatus(connected) {
  const el = document.getElementById('ws-status');
  const dots = document.querySelectorAll('#system-status-dot, #st-ws');
  if (connected) {
    el.textContent = 'Connected';
    el.className = 'connected';
    document.getElementById('sys-status-label').textContent = 'Online';
    document.getElementById('st-ws').className = 'status-pill online';
    document.getElementById('st-ws').textContent = 'online';
  } else {
    el.textContent = 'Disconnected';
    el.className = 'disconnected';
    document.getElementById('sys-status-label').textContent = 'Offline';
    document.getElementById('st-ws').className = 'status-pill offline';
    document.getElementById('st-ws').textContent = 'offline';
  }
}

// ─────────────────────────────────────────────────────────
// DASHBOARD UPDATER  (called on every WS message)
// ─────────────────────────────────────────────────────────
function updateDashboard(data) {
  const det   = data.detection  || {};
  const sumo  = data.sumo       || {};
  const marl  = data.marl       || {};
  const inter = data.intersections || [];
  const emerg = data.emergency  || {};
  const hist  = data.history    || {};
  const sys   = data.system     || {};

  // ── Server clock ─────────────────────────────────────
  if (sys.server_time) {
    document.getElementById('server-clock').textContent = sys.server_time;
  }

  // ── Detection KPIs ────────────────────────────────────
  setText('kpi-total',   det.vehicle_count    ?? 0);
  setText('kpi-cars',    det.car_count        ?? 0);
  setText('kpi-moto',    det.motorcycle_count ?? 0);
  setText('kpi-buses',   det.bus_count        ?? 0);
  setText('kpi-trucks',  det.truck_count      ?? 0);
  setText('kpi-density', (det.density ?? 0) + '%');
  setText('kpi-gnn',     det.gnn_prediction ?? 'LOW');
  setText('kpi-detection', det.detection_active ? 'LIVE' : 'Ready');

  // ── Traffic level badge ───────────────────────────────
  const level = det.traffic_level || 'LOW';
  const badge = document.getElementById('traffic-level-badge');
  if (badge) {
    badge.textContent = level;
    badge.className   = `traffic-level-badge ${level}`;
  }

  // ── Density progress bar ─────────────────────────────
  const density = det.density ?? 0;
  setText('density-pct', density + '%');
  const bar = document.getElementById('density-bar');
  if (bar) {
    bar.style.width = density + '%';
    bar.style.background =
      density >= 65 ? 'var(--red)' :
      density >= 35 ? 'var(--yellow)' :
      'var(--green)';
  }

  // ── Overlay on video ─────────────────────────────────
  setText('ov-frame', `Frame: ${det.frame_number ?? 0}`);
  setText('ov-level', `Level: ${level}`);
  setText('gnn-detail', det.gnn_prediction ?? 'LOW');

  // ── Video feed ───────────────────────────────────────
  if (det.frame_b64) {
    const src = `data:image/jpeg;base64,${det.frame_b64}`;
    updateVideoFeed('video-feed', 'video-placeholder', src);
    updateVideoFeed('video-feed-2', 'video-placeholder-2', src);
    setText('feed-badge', 'LIVE');
    setText('live-feed-badge', 'LIVE');
    document.getElementById('feed-badge').className = 'badge badge-live';
  } else {
    setText('feed-badge', 'OFFLINE');
    document.getElementById('feed-badge').className = 'badge badge-stub';
  }

  // ── YOLO status pill ─────────────────────────────────
  const yoloPill = document.getElementById('st-yolo');
  if (yoloPill) {
    if (sys.detection_running) {
      yoloPill.className   = 'status-pill online';
      yoloPill.textContent = 'live';
    } else {
      yoloPill.className   = 'status-pill offline';
      yoloPill.textContent = 'offline';
    }
  }

  // ── Detection list ───────────────────────────────────
  updateDetectionList(det.detections || []);

  // ── Vehicle type legend ──────────────────────────────
  setText('legend-cars',   det.car_count        ?? 0);
  setText('legend-moto',   det.motorcycle_count ?? 0);
  setText('legend-buses',  det.bus_count        ?? 0);
  setText('legend-trucks', det.truck_count      ?? 0);

  // ── Intersections grid ───────────────────────────────
  renderIntersections(inter);

  // ── MARL panel ───────────────────────────────────────
  setText('marl-agents',        marl.num_agents            ?? 0);
  setText('marl-intersections', marl.num_intersections     ?? 0);
  setText('marl-episode',       marl.training_episode      ?? 0);
  setText('marl-reward',        marl.current_reward ?? 'N/A');
  setText('marl-version',       `v${marl.global_model_version ?? 0}`);
  setText('marl-action',        marl.latest_rl_action      ?? 'None');
  setText('marl-round',         marl.training_round        ?? 0);

  // ── SUMO panel ────────────────────────────────────────
  setText('sumo-status',   sumo.connected ? 'Connected' : 'Disconnected');
  setText('sumo-time',     (sumo.sim_time ?? 0) + 's');
  setText('sumo-vehicles', sumo.vehicle_count ?? 0);
  setText('sumo-wait',     (sumo.avg_wait_time ?? 0) + 's');

  // ── Emergency panel ───────────────────────────────────
  setText('emerg-detected',     emerg.detected ? 'YES ⚠️' : 'No');
  setText('emerg-type',         emerg.vehicle_type         ?? '—');
  setText('emerg-location',     emerg.location             ?? '—');
  setText('emerg-time',         emerg.detection_time       ?? '—');
  setText('emerg-priority',     emerg.priority_status      ?? '—');
  setText('emerg-intersection', emerg.affected_intersection?? '—');
  setText('emerg-signal',       emerg.signal_priority ? 'YES 🔴' : 'No');

  // ── Charts ───────────────────────────────────────────
  if (hist.timestamps && hist.timestamps.length > 0) {
    updateCountChart(hist);
    updateDensityChart(hist);
    updateLevelChart(hist);
    updateFlowChart(hist);
    updateHistoryChart(hist);
    updateDonutChart(det);
  }

  // ── Start/Stop button state ───────────────────────────
  const btnStart = document.getElementById('btn-start-det');
  const btnStop  = document.getElementById('btn-stop-det');
  if (btnStart && btnStop) {
    btnStart.disabled = sys.detection_running;
    btnStop.disabled  = !sys.detection_running;
  }
}

// ─────────────────────────────────────────────────────────
// VIDEO FEED HELPER
// ─────────────────────────────────────────────────────────
function updateVideoFeed(imgId, placeholderId, src) {
  const img = document.getElementById(imgId);
  const ph  = document.getElementById(placeholderId);
  if (!img) return;
  img.src = src;
  img.style.display = 'block';
  if (ph) ph.style.display = 'none';
}

// ─────────────────────────────────────────────────────────
// DETECTION LIST
// ─────────────────────────────────────────────────────────
function updateDetectionList(detections) {
  const el = document.getElementById('detection-list');
  if (!el) return;
  if (!detections || detections.length === 0) {
    el.innerHTML = '<p style="color:var(--gray);font-size:12px;">No vehicles in ROI.</p>';
    return;
  }
  const rows = detections.map(d =>
    `<div style="display:flex;justify-content:space-between;align-items:center;
                 padding:6px 0;border-bottom:1px solid var(--border);font-size:12px;">
       <span>${vehicleEmoji(d.type)} ${d.type}</span>
       <span style="color:var(--gray);">ID: ${d.id ?? '—'}</span>
       <span style="color:var(--cyan);">${d.conf.toFixed(2)}</span>
     </div>`
  ).join('');
  el.innerHTML = rows;
}

function vehicleEmoji(type) {
  return {Car:'🚗', Motorcycle:'🏍️', Bus:'🚌', Truck:'🚚'}[type] || '🚗';
}

// ─────────────────────────────────────────────────────────
// INTERSECTIONS GRID
// ─────────────────────────────────────────────────────────
function renderIntersections(intersections) {
  const grid = document.getElementById('intersections-grid');
  if (!grid) return;
  if (intersections.length === 0) return;
  grid.innerHTML = intersections.map(inter => {
    const sig = (inter.signal || 'RED').toUpperCase();
    return `
      <div class="intersection-card">
        <div class="intersection-header">
          <div class="intersection-title">🚦 Intersection ${inter.id}</div>
          <span class="status-pill stub">NOT IMPL.</span>
        </div>
        <div class="intersection-body">
          <div class="signal-group">
            <div class="signal-lamp red  ${sig==='RED'   ? 'active':''}"></div>
            <div class="signal-lamp yellow ${sig==='YELLOW'? 'active':''}"></div>
            <div class="signal-lamp green ${sig==='GREEN' ? 'active':''}"></div>
          </div>
          <div class="intersection-info">
            <div class="info-row">
              <span class="info-label">Signal</span>
              <span class="info-value">${sig}</span>
            </div>
            <div class="info-row">
              <span class="info-label">Remaining</span>
              <span class="info-value">${inter.remaining_time ?? 0}s</span>
            </div>
            <div class="info-row">
              <span class="info-label">Density</span>
              <span class="info-value">${inter.density ?? 0}%</span>
            </div>
            <div class="info-row">
              <span class="info-label">Queue</span>
              <span class="info-value">${inter.queue_length ?? 0}</span>
            </div>
            <div class="info-row">
              <span class="info-label">RL Action</span>
              <span class="info-value">${inter.rl_action ?? '—'}</span>
            </div>
            <div class="info-row">
              <span class="info-label">Next Phase</span>
              <span class="info-value">${inter.next_phase ?? '—'}</span>
            </div>
          </div>
        </div>
      </div>`;
  }).join('');
}

// ─────────────────────────────────────────────────────────
// CHARTS
// ─────────────────────────────────────────────────────────
const chartDefaults = {
  animation:   { duration: 0 },
  responsive:  true,
  maintainAspectRatio: false,
  plugins: {
    legend: { labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } } },
    tooltip: {
      backgroundColor: '#111e3a',
      borderColor: '#1e2f56',
      borderWidth: 1,
      titleColor: '#f1f5f9',
      bodyColor: '#cbd5e1',
    },
  },
  scales: {
    x: {
      ticks: { color: '#64748b', font: { family: 'Inter', size: 10 }, maxTicksLimit: 8 },
      grid:  { color: 'rgba(30,47,86,0.6)' },
    },
    y: {
      ticks: { color: '#64748b', font: { family: 'Inter', size: 10 } },
      grid:  { color: 'rgba(30,47,86,0.6)' },
    },
  },
};

function makeLineChart(canvasId, label, color, maxY) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label,
        data: [],
        borderColor: color,
        backgroundColor: color + '22',
        borderWidth: 2,
        pointRadius: 2,
        fill: true,
        tension: 0.4,
      }],
    },
    options: {
      ...chartDefaults,
      scales: {
        ...chartDefaults.scales,
        y: { ...chartDefaults.scales.y, max: maxY, min: 0 },
      },
    },
  });
}

function updateCountChart(hist) {
  if (!chartCount) {
    chartCount = makeLineChart('chart-count', 'Vehicles', '#3b82f6', null);
  }
  if (!chartCount) return;
  chartCount.data.labels = hist.timestamps;
  chartCount.data.datasets[0].data = hist.vehicle_counts;
  chartCount.update('none');
}

function updateDensityChart(hist) {
  if (!chartDensity) {
    chartDensity = makeLineChart('chart-density', 'Density %', '#06b6d4', 100);
  }
  if (!chartDensity) return;
  chartDensity.data.labels = hist.timestamps;
  chartDensity.data.datasets[0].data = hist.densities;
  chartDensity.update('none');
}

function updateLevelChart(hist) {
  if (!chartLevel) {
    const ctx = document.getElementById('chart-level');
    if (!ctx) return;
    chartLevel = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Traffic Level (0=LOW,1=MED,2=HIGH)',
          data: [],
          borderColor: '#f59e0b',
          backgroundColor: '#f59e0b22',
          borderWidth: 2, pointRadius: 2, fill: true, tension: 0.4,
        }],
      },
      options: {
        ...chartDefaults,
        scales: {
          ...chartDefaults.scales,
          y: { ...chartDefaults.scales.y, min: 0, max: 2,
               ticks: { ...chartDefaults.scales.y.ticks, callback: v => ['LOW','MED','HIGH'][v] ?? v } },
        },
      },
    });
  }
  const encoded = (hist.traffic_levels || []).map(l => l==='HIGH'?2:l==='MEDIUM'?1:0);
  chartLevel.data.labels = hist.timestamps;
  chartLevel.data.datasets[0].data = encoded;
  chartLevel.update('none');
}

function updateFlowChart(hist) {
  if (!chartFlow) {
    chartFlow = makeLineChart('chart-flow', 'Traffic Flow', '#10b981', null);
  }
  if (!chartFlow) return;
  chartFlow.data.labels = hist.timestamps;
  chartFlow.data.datasets[0].data = hist.vehicle_counts;
  chartFlow.update('none');
}

function updateHistoryChart(hist) {
  if (!chartHistory) {
    chartHistory = makeLineChart('chart-history', 'Vehicle Count', '#7c3aed', null);
  }
  if (!chartHistory) return;
  chartHistory.data.labels = hist.timestamps;
  chartHistory.data.datasets[0].data = hist.vehicle_counts;
  chartHistory.update('none');
}

function updateDonutChart(det) {
  const ctx = document.getElementById('chart-donut');
  if (!ctx) return;
  const values = [
    det.car_count ?? 0,
    det.motorcycle_count ?? 0,
    det.bus_count ?? 0,
    det.truck_count ?? 0,
  ];
  if (!chartDonut) {
    chartDonut = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Cars', 'Motorcycles', 'Buses', 'Trucks'],
        datasets: [{
          data: values,
          backgroundColor: ['#3b82f6','#7c3aed','#f59e0b','#ec4899'],
          borderWidth: 0,
          hoverOffset: 6,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        cutout: '65%',
        plugins: {
          legend: { display: false },
          tooltip: chartDefaults.plugins.tooltip,
        },
        animation: { duration: 300 },
      },
    });
  } else {
    chartDonut.data.datasets[0].data = values;
    chartDonut.update('none');
  }
}

// Reward & Phase charts (stubs — no data yet)
function initStubCharts() {
  const ctxReward = document.getElementById('chart-reward');
  if (ctxReward && !chartReward) {
    chartReward = new Chart(ctxReward, {
      type: 'line',
      data: { labels: [], datasets: [{
        label: 'RL Reward', data: [],
        borderColor: '#10b981', backgroundColor: '#10b98122',
        borderWidth: 2, pointRadius: 2, fill: true, tension: 0.4,
      }]},
      options: { ...chartDefaults },
    });
  }

  const ctxPhase = document.getElementById('chart-phase');
  if (ctxPhase && !chartPhase) {
    chartPhase = new Chart(ctxPhase, {
      type: 'bar',
      data: {
        labels: ['Intersection 1', 'Intersection 2', 'Intersection 3'],
        datasets: [
          { label: 'Green', data: [0, 0, 0], backgroundColor: '#10b981' },
          { label: 'Yellow', data: [0, 0, 0], backgroundColor: '#f59e0b' },
          { label: 'Red',   data: [0, 0, 0], backgroundColor: '#ef4444' },
        ],
      },
      options: { ...chartDefaults, scales: {
        ...chartDefaults.scales,
        x: { ...chartDefaults.scales.x, stacked: true },
        y: { ...chartDefaults.scales.y, stacked: true },
      }},
    });
  }
}

// ─────────────────────────────────────────────────────────
// DETECTION CONTROL
// ─────────────────────────────────────────────────────────
async function startDetection() {
  try {
    const res = await fetch(`${API_BASE}/api/detection/start`, { method: 'POST' });
    const data = await res.json();
    addLog('INFO', `Detection start: ${data.status}`);
  } catch (e) {
    addLog('ERROR', `Could not start detection: ${e.message}`);
  }
}

async function stopDetection() {
  try {
    const res = await fetch(`${API_BASE}/api/detection/stop`, { method: 'POST' });
    const data = await res.json();
    addLog('INFO', `Detection stop: ${data.status}`);
    // Clear video
    ['video-feed','video-feed-2'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.style.display = 'none';
    });
    ['video-placeholder','video-placeholder-2'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.style.display = '';
    });
  } catch (e) {
    addLog('ERROR', `Could not stop detection: ${e.message}`);
  }
}

async function sumoControl(action) {
  try {
    const res = await fetch(`${API_BASE}/api/sumo/${action}`, { method: 'POST' });
    const data = await res.json();
    addLog('INFO', `SUMO ${action}: ${data.status}`);
  } catch (e) {
    addLog('ERROR', `SUMO control error: ${e.message}`);
  }
}

// ─────────────────────────────────────────────────────────
// LOG SYSTEM
// ─────────────────────────────────────────────────────────
function addLog(level, message) {
  const ts = new Date().toLocaleTimeString();
  logBuffer.unshift({ ts, level, message });
  if (logBuffer.length > 200) logBuffer.pop();
  renderLogs();
}

function renderLogs() {
  const tbody = document.getElementById('log-body');
  if (!tbody) return;
  const colorMap = { INFO: '#10b981', WARN: '#f59e0b', ERROR: '#ef4444' };
  tbody.innerHTML = logBuffer.slice(0, 50).map(l =>
    `<tr>
       <td>${l.ts}</td>
       <td style="color:${colorMap[l.level]||'#fff'};font-weight:700;">${l.level}</td>
       <td>${l.message}</td>
     </tr>`
  ).join('') || '<tr><td colspan="3" style="color:var(--gray);text-align:center;">No logs.</td></tr>';
}

function clearLogs() {
  logBuffer = [];
  renderLogs();
}

// ─────────────────────────────────────────────────────────
// UTILITY
// ─────────────────────────────────────────────────────────
function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

// ─────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Clock fallback (server clock takes over when WS connects)
  setInterval(() => {
    const el = document.getElementById('server-clock');
    if (el && (!lastState || !lastState.system)) {
      el.textContent = new Date().toLocaleTimeString();
    }
  }, 1000);

  initStubCharts();
  renderIntersections([
    { id:1, signal:'RED',   remaining_time:0, density:0, queue_length:0, rl_action:'—', next_phase:'—' },
    { id:2, signal:'GREEN', remaining_time:0, density:0, queue_length:0, rl_action:'—', next_phase:'—' },
    { id:3, signal:'RED',   remaining_time:0, density:0, queue_length:0, rl_action:'—', next_phase:'—' },
  ]);

  addLog('INFO', 'Dashboard loaded. Connecting to backend…');
  connectWS();
});
