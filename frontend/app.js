// Real-Time Transaction Fraud Detection - Operator Dashboard Logic
(function () {
  'use strict';

  // State Management
  const state = {
    isPaused: false,
    selectedTx: null,
    metricsMetadata: null,
    activeThreshold: 0.15,
    kpis: {
      totalProcessed: 0,
      flagged: 0,
      flaggedAmount: 0.0,
      totalLatency: 0.0,
    },
    transactions: [],
  };

  // API key (only needed when the server runs with APP_ENV=production)
  const API_KEY_STORAGE = 'fraudApiKey';

  function authHeaders(headers = {}) {
    const key = localStorage.getItem(API_KEY_STORAGE);
    return key ? { ...headers, 'X-API-Key': key } : headers;
  }

  function promptForApiKey() {
    const key = window.prompt('This API requires an X-API-Key. Paste it to enable dashboard controls:');
    if (key) localStorage.setItem(API_KEY_STORAGE, key.trim());
  }

  // Stream fields are server-generated, but the feed must never treat them as markup.
  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, ch => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[ch]));
  }

  // DOM Elements
  const el = {
    connectionStatus: document.getElementById('connection-status'),
    connectionText: document.getElementById('connection-text'),
    championBadge: document.getElementById('champion-badge'),
    kpiTotalTx: document.getElementById('kpi-total-tx'),
    kpiInterceptRate: document.getElementById('kpi-intercept-rate'),
    kpiDollarsSaved: document.getElementById('kpi-dollars-saved'),
    kpiThreshold: document.getElementById('kpi-threshold'),
    kpiLatency: document.getElementById('kpi-latency'),
    feedList: document.getElementById('feed-list'),
    btnPauseFeed: document.getElementById('btn-pause-feed'),
    btnClearFeed: document.getElementById('btn-clear-feed'),
    inspectId: document.getElementById('inspect-id'),
    inspectUser: document.getElementById('inspect-user'),
    inspectAmount: document.getElementById('inspect-amount'),
    inspectProb: document.getElementById('inspect-prob'),
    inspectDecisionBadge: document.getElementById('inspect-decision-badge'),
    inspectLatency: document.getElementById('inspect-latency'),
    shapBarsList: document.getElementById('shap-bars-list'),
    reasonPillsList: document.getElementById('reason-pills-list'),
    sliderThreshold: document.getElementById('threshold-range-slider'),
    sliderThresholdVal: document.getElementById('slider-threshold-val'),
    simFnCost: document.getElementById('sim-fn-cost'),
    simFpCost: document.getElementById('sim-fp-cost'),
    simTpPrevented: document.getElementById('sim-tp-prevented'),
    simTotalLoss: document.getElementById('sim-total-loss'),
    benchmarkTbody: document.getElementById('benchmark-tbody'),
    benchmarkSubtitle: document.getElementById('benchmark-subtitle'),
    prCurveSvg: document.getElementById('pr-curve-svg'),
    btnAttackCardTesting: document.getElementById('btn-attack-card-testing'),
    btnAttackTakeover: document.getElementById('btn-attack-takeover'),
    btnAttackNormal: document.getElementById('btn-attack-normal'),
  };

  // 1. Tab Switching
  const tabButtons = Array.from(document.querySelectorAll('.tab-btn'));

  function activateTab(btn) {
    tabButtons.forEach(tab => {
      const active = tab === btn;
      tab.classList.toggle('active', active);
      tab.setAttribute('aria-selected', active ? 'true' : 'false');
      tab.tabIndex = active ? 0 : -1;
      const panel = document.getElementById(`tab-${tab.dataset.tab}`);
      if (panel) panel.style.display = active ? 'block' : 'none';
    });
  }

  tabButtons.forEach((btn, index) => {
    btn.addEventListener('click', () => activateTab(btn));
    btn.addEventListener('keydown', evt => {
      let next = null;
      if (evt.key === 'ArrowRight') next = tabButtons[(index + 1) % tabButtons.length];
      if (evt.key === 'ArrowLeft') next = tabButtons[(index - 1 + tabButtons.length) % tabButtons.length];
      if (evt.key === 'Home') next = tabButtons[0];
      if (evt.key === 'End') next = tabButtons[tabButtons.length - 1];
      if (next) {
        evt.preventDefault();
        activateTab(next);
        next.focus();
      }
    });
  });

  // 2. Fetch Initial Metrics and Benchmarks
  async function loadMetrics() {
    try {
      const res = await fetch('/metrics');
      if (!res.ok) throw new Error(`metrics returned ${res.status}`);
      const data = await res.json();
      state.metricsMetadata = data;

      if (el.championBadge && data.champion_algorithm) {
        const prAuc = Number(data.pr_auc);
        el.championBadge.innerText = Number.isFinite(prAuc)
          ? `${data.champion_algorithm} Champion (PR-AUC ${prAuc.toFixed(4)})`
          : `${data.champion_algorithm} Champion`;
      }
      if (el.benchmarkSubtitle && Number.isFinite(Number(data.test_fraud_rate))) {
        el.benchmarkSubtitle.innerText =
          `Model benchmark on ${(Number(data.test_fraud_rate) * 100).toFixed(2)}% fraud test set` +
          (Number.isFinite(Number(data.test_size))
            ? ` (n=${Number(data.test_size).toLocaleString()})`
            : '');
      }

      renderBenchmarks(data);
      renderCostSimulator(state.activeThreshold);
    } catch (err) {
      console.warn('Metrics endpoint unavailable; benchmark and cost panels stay empty.');
      renderCostSimulator(state.activeThreshold);
    }
  }

  // Keep the dashboard on the thresholds the API is actually enforcing.
  async function loadThresholds() {
    try {
      const res = await fetch('/threshold');
      if (!res.ok) return;
      const config = await res.json();
      applyThreshold(config.decline_threshold);
    } catch (err) {
      console.warn('Threshold endpoint unavailable, keeping the built-in default.');
    }
  }

  function applyThreshold(val) {
    state.activeThreshold = val;
    if (el.sliderThreshold) {
      el.sliderThreshold.value = val;
      el.sliderThreshold.setAttribute('aria-valuetext', `${val.toFixed(2)} decline threshold`);
    }
    if (el.sliderThresholdVal) el.sliderThresholdVal.innerText = val.toFixed(2);
    if (el.kpiThreshold) el.kpiThreshold.innerText = val.toFixed(2);
    renderCostSimulator(val);
  }

  function renderBenchmarks(data) {
    if (!data || !data.benchmark_comparison || !el.benchmarkTbody) return;
    const metric = value =>
      Number.isFinite(Number(value)) ? Number(value).toFixed(4) : '&mdash;';

    el.benchmarkTbody.innerHTML = '';
    data.benchmark_comparison.forEach(m => {
      const tr = document.createElement('tr');
      tr.style.borderBottom = '1px solid rgba(255, 255, 255, 0.05)';
      const isChamp =
        Number(m.rank) === 1 ||
        (data.champion_algorithm &&
          String(m.model_name).toLowerCase().includes(String(data.champion_algorithm).toLowerCase()));
      tr.innerHTML = `
        <td style="padding: 10px 4px; font-weight: 600; color: ${isChamp ? '#38bdf8' : '#ffffff'};">
          ${isChamp ? '🏆 ' : ''}${escapeHtml(m.model_name)}
        </td>
        <td style="padding: 10px 4px; font-family: monospace; font-weight: 700; color: #a7f3d0;">${metric(m.pr_auc)}</td>
        <td style="padding: 10px 4px; font-family: monospace;">${metric(m.roc_auc)}</td>
        <td style="padding: 10px 4px; font-family: monospace;">${metric(m.precision)}</td>
        <td style="padding: 10px 4px; font-family: monospace;">${metric(m.recall)}</td>
      `;
      el.benchmarkTbody.appendChild(tr);
    });

    // Render PR-AUC Curve SVG with the operating point from the served threshold.
    if (data.pr_curve && el.prCurveSvg) {
      const { precision, recall, thresholds } = data.pr_curve;
      if (!Array.isArray(precision) || !Array.isArray(recall) || !precision.length) return;

      let pathD = '';
      precision.forEach((p, idx) => {
        const r = recall[idx];
        const x = 30 + r * 340;
        const y = 160 - p * 140;
        pathD += `${idx === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)} `;
      });

      let opIndex = 0;
      const target = Number(data.optimal_threshold);
      if (Array.isArray(thresholds) && Number.isFinite(target)) {
        thresholds.forEach((t, idx) => {
          if (Math.abs(t - target) < Math.abs(thresholds[opIndex] - target)) opIndex = idx;
        });
      }
      const opX = 30 + recall[opIndex] * 340;
      const opY = 160 - precision[opIndex] * 140;
      const labelX = opX > 300 ? opX - 92 : opX + 10;
      const labelY = opY < 34 ? opY + 16 : opY - 6;
      const opThreshold = Array.isArray(thresholds) ? thresholds[opIndex] : null;
      const opText = Number.isFinite(Number(opThreshold))
        ? `threshold ${Number(opThreshold).toFixed(2)}`
        : 'the cost-optimal threshold';

      el.prCurveSvg.setAttribute(
        'aria-label',
        `Precision-recall curve for ${data.champion_algorithm || 'the champion model'}, operating point at ${opText}`
      );
      el.prCurveSvg.innerHTML = `
        <line x1="30" y1="20" x2="30" y2="160" stroke="#475569" stroke-width="1"/>
        <line x1="30" y1="160" x2="380" y2="160" stroke="#475569" stroke-width="1"/>
        <text x="35" y="32" fill="#94a3b8" font-size="10">Precision (1.0)</text>
        <text x="320" y="152" fill="#94a3b8" font-size="10">Recall (1.0)</text>
        <path d="${pathD}" fill="none" stroke="#06b6d4" stroke-width="2.5"/>
        <circle cx="${opX.toFixed(1)}" cy="${opY.toFixed(1)}" r="4.5" fill="#ef4444" stroke="#ffffff" stroke-width="1.5"/>
        <text x="${labelX.toFixed(1)}" y="${labelY.toFixed(1)}" fill="#fca5a5" font-size="10" font-weight="bold">Operating Point &tau;*</text>
      `;
    }
  }

  // 3. Cost-Utility Simulator
  function renderCostSimulator(threshold) {
    const nodes = [el.simFnCost, el.simFpCost, el.simTpPrevented, el.simTotalLoss];
    const curve = state.metricsMetadata && state.metricsMetadata.threshold_curve;
    if (!Array.isArray(curve) || !curve.length) {
      nodes.forEach(node => { if (node) node.innerText = '—'; });
      return;
    }

    // Find closest threshold item
    let closest = curve[0];
    curve.forEach(item => {
      if (Math.abs(item.threshold - threshold) < Math.abs(closest.threshold - threshold)) {
        closest = item;
      }
    });

    const money = value =>
      Number.isFinite(Number(value))
        ? `$${Number(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
        : '—';
    if (el.simFnCost) el.simFnCost.innerText = money(closest.missed_fraud_dollar_loss);
    if (el.simFpCost) el.simFpCost.innerText = money(closest.false_positive_friction_cost);
    if (el.simTpPrevented) el.simTpPrevented.innerText = money(closest.fraud_dollars_prevented);
    if (el.simTotalLoss) el.simTotalLoss.innerText = money(closest.total_financial_loss);
  }

  let thresholdTimer = null;
  if (el.sliderThreshold) {
    el.sliderThreshold.addEventListener('input', e => {
      const val = parseFloat(e.target.value);
      applyThreshold(val);
      // Debounce: dragging must not spend the client's rate-limit budget.
      clearTimeout(thresholdTimer);
      thresholdTimer = setTimeout(() => {
        fetch('/threshold', {
          method: 'POST',
          headers: authHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({ approve_threshold: Math.max(0.01, val * 0.5), decline_threshold: val }),
        })
          .then(res => {
            if (res.status === 401) promptForApiKey();
          })
          .catch(() => {});
      }, 250);
    });
  }

  // 4. WebSocket Live Stream
  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || 'localhost:8000';
    const wsUrl = `${protocol}//${host}/ws/transactions`;

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      el.connectionStatus.className = 'status-badge live';
      el.connectionText.innerText = 'STREAM ACTIVE (LIVE)';
    };

    ws.onmessage = evt => {
      if (state.isPaused) return;
      try {
        const event = JSON.parse(evt.data);
        handleTransactionEvent(event);
      } catch (err) {
        console.error('Error parsing transaction event:', err);
      }
    };

    ws.onclose = () => {
      el.connectionStatus.className = 'status-badge';
      el.connectionText.innerText = 'RECONNECTING STREAM...';
      setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }

  // 5. Transaction Event Processing
  function selectRow(row, tx) {
    document.querySelectorAll('.tx-row').forEach(other => {
      other.classList.remove('selected');
      other.removeAttribute('aria-current');
    });
    row.classList.add('selected');
    row.setAttribute('aria-current', 'true');
    selectTransactionForInspector(tx);
  }

  function handleTransactionEvent(tx) {
    state.transactions.unshift(tx);
    if (state.transactions.length > 50) state.transactions.pop();

    // KPIs count flagged volume, not confirmed fraud: the stream carries no label.
    const amount = Number(tx.Amount) || 0;
    state.kpis.totalProcessed += 1;
    state.kpis.totalLatency += Number(tx.latency_ms) || 0;
    const isFlagged = tx.decision === 'DECLINE' || tx.decision === 'MANUAL_REVIEW';
    if (isFlagged) {
      state.kpis.flagged += 1;
      state.kpis.flaggedAmount += amount;
    }

    el.kpiTotalTx.innerText = state.kpis.totalProcessed.toLocaleString();
    const rate = (state.kpis.flagged / state.kpis.totalProcessed) * 100;
    el.kpiInterceptRate.innerText = `${rate.toFixed(1)}%`;
    el.kpiDollarsSaved.innerText = `$${state.kpis.flaggedAmount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    const avgLatency = state.kpis.totalLatency / state.kpis.totalProcessed;
    el.kpiLatency.innerText = `${avgLatency.toFixed(1)}ms`;

    // Render row in feed
    const row = document.createElement('div');
    row.className = 'tx-row';
    row.tabIndex = 0;
    row.setAttribute('role', 'button');
    row.setAttribute(
      'aria-label',
      `Transaction ${tx.transaction_id}, $${amount.toFixed(2)}, ${tx.decision}, ` +
      `${(Number(tx.fraud_probability) * 100).toFixed(1)} percent fraud probability`
    );
    const badgeClass =
      tx.decision === 'APPROVE' ? 'badge-approve' : tx.decision === 'MANUAL_REVIEW' ? 'badge-review' : 'badge-decline';

    row.innerHTML = `
      <div class="tx-id">${escapeHtml(tx.transaction_id)}</div>
      <div class="tx-amount">$${amount.toFixed(2)}</div>
      <div class="tx-user">${escapeHtml(tx.user_id)}</div>
      <div><span class="tx-decision-badge ${badgeClass}">${escapeHtml(tx.decision)}</span></div>
      <div class="tx-prob" style="color: ${tx.fraud_probability > 0.5 ? '#f87171' : '#a7f3d0'};">
        ${(Number(tx.fraud_probability) * 100).toFixed(1)}%
      </div>
    `;

    row.addEventListener('click', () => selectRow(row, tx));
    row.addEventListener('keydown', evt => {
      if (evt.key === 'Enter' || evt.key === ' ') {
        evt.preventDefault();
        selectRow(row, tx);
      }
    });

    el.feedList.insertBefore(row, el.feedList.firstChild);
    if (el.feedList.children.length > 50) {
      el.feedList.removeChild(el.feedList.lastChild);
    }

    // Auto-inspect if no transaction selected or if high fraud alert occurs
    if (!state.selectedTx || (tx.decision === 'DECLINE' && state.selectedTx.decision === 'APPROVE')) {
      selectRow(row, tx);
    }
  }

  // 6. Inspector & TreeSHAP Rendering
  function selectTransactionForInspector(tx) {
    state.selectedTx = tx;
    const amount = Number(tx.Amount) || 0;
    el.inspectId.innerText = tx.transaction_id;
    el.inspectUser.innerText = tx.user_id;
    el.inspectAmount.innerText = `$${amount.toFixed(2)}`;
    el.inspectProb.innerText = `${(Number(tx.fraud_probability) * 100).toFixed(1)}%`;
    el.inspectLatency.innerText = `${(Number(tx.latency_ms) || 0).toFixed(1)}ms`;

    const badgeClass =
      tx.decision === 'APPROVE' ? 'badge-approve' : tx.decision === 'MANUAL_REVIEW' ? 'badge-review' : 'badge-decline';
    el.inspectDecisionBadge.className = `tx-decision-badge ${badgeClass}`;
    el.inspectDecisionBadge.innerText = tx.decision;

    // Render SHAP waterfall bars
    el.shapBarsList.innerHTML = '';
    el.reasonPillsList.innerHTML = '';

    if (tx.explanation && tx.explanation.top_risk_factors && tx.explanation.top_risk_factors.length) {
      const riskFactors = tx.explanation.top_risk_factors;
      const maxImpact = Math.max(0.1, ...riskFactors.map(rf => Math.abs(rf.shap_impact)));

      riskFactors.forEach(rf => {
        const row = document.createElement('div');
        row.className = 'shap-row';
        const pctWidth = Math.min(100, (Math.abs(rf.shap_impact) / maxImpact) * 100);
        const displayValue = rf.feature_value_display ?? rf.feature_value;
        row.innerHTML = `
          <div class="shap-row-header">
            <span style="color: #ffffff;">${escapeHtml(rf.feature_name)}: ${escapeHtml(displayValue)}</span>
            <span style="font-family: monospace; color: #f87171;">+${rf.shap_impact.toFixed(2)} risk</span>
          </div>
          <div class="shap-bar-bg">
            <div class="shap-bar-fill risk" style="width: ${pctWidth}%;"></div>
          </div>
        `;
        el.shapBarsList.appendChild(row);

        // Model attribution, not a regulator-approved adverse action code.
        const pill = document.createElement('div');
        pill.className = 'reason-pill';
        pill.innerText = `Risk reason: ${rf.reason_code}`;
        el.reasonPillsList.appendChild(pill);
      });
    } else {
      el.shapBarsList.innerHTML = `<div style="color: var(--text-dim); font-size: 0.8rem;">No material risk factors for this transaction.</div>`;
    }
  }

  // 7. Feed Control Buttons
  if (el.btnPauseFeed) {
    el.btnPauseFeed.addEventListener('click', () => {
      state.isPaused = !state.isPaused;
      el.btnPauseFeed.innerHTML = state.isPaused ? '&#x25B6; Resume' : '&#x23F8; Pause';
      el.btnPauseFeed.classList.toggle('active', state.isPaused);
      el.btnPauseFeed.setAttribute('aria-pressed', String(state.isPaused));
      el.btnPauseFeed.setAttribute('aria-label', state.isPaused ? 'Resume live feed' : 'Pause live feed');
    });
  }

  if (el.btnClearFeed) {
    el.btnClearFeed.addEventListener('click', () => {
      el.feedList.innerHTML = '';
      state.transactions = [];
    });
  }

  // 8. Attack Simulation Triggers
  async function triggerAttackScenario(scenario, numEvents) {
    try {
      const res = await fetch('/stream/attack', {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ scenario, num_events: numEvents }),
      });
      if (res.status === 401) {
        promptForApiKey();
        return;
      }
      const data = await res.json();
      console.log('Attack Trigger Response:', data);
    } catch (err) {
      console.error('Failed to trigger attack scenario:', err);
    }
  }

  if (el.btnAttackCardTesting) {
    el.btnAttackCardTesting.addEventListener('click', () => triggerAttackScenario('card_testing', 20));
  }
  if (el.btnAttackTakeover) {
    el.btnAttackTakeover.addEventListener('click', () => triggerAttackScenario('account_takeover', 12));
  }
  if (el.btnAttackNormal) {
    el.btnAttackNormal.addEventListener('click', () => triggerAttackScenario('normal_traffic', 15));
  }

  // Initialize
  if (tabButtons.length) activateTab(tabButtons.find(tab => tab.classList.contains('active')) || tabButtons[0]);
  loadMetrics();
  loadThresholds();
  connectWebSocket();
})();
