// SFM-GNN+ Real-Time SOC Interface Client
document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const kpiTotal = document.getElementById("kpiTotal");
  const kpiAttacks = document.getElementById("kpiAttacks");
  const kpiSuppressed = document.getElementById("kpiSuppressed");
  const kpiClean = document.getElementById("kpiClean");
  const quarantineCount = document.getElementById("quarantineCount");
  const modeSubText = document.getElementById("modeSubText");
  const modeSelect = document.getElementById("modeSelect");

  const streamToggleBtn = document.getElementById("streamToggleBtn");
  const deviceName = document.getElementById("deviceName");
  const eventsContainer = document.getElementById("eventsContainer");
  const quarantineTableBody = document.getElementById("quarantineTableBody");
  
  const xaiTargetClass = document.getElementById("xaiTargetClass");
  const xaiBars = document.getElementById("xaiBars");
  
  const canvas = document.getElementById("threatGraphCanvas");
  const ctx = canvas.getContext("2d");
  const graphOverlay = document.getElementById("graphOverlay");

  // Federated Learning Elements
  const runFedRoundBtn = document.getElementById("runFedRoundBtn");
  const testPoisonDefenseBtn = document.getElementById("testPoisonDefenseBtn");
  const secaggRoundText = document.getElementById("secaggRoundText");
  const secaggDetailText = document.getElementById("secaggDetailText");

  let isStreaming = true;
  let socket = null;
  let currentNodes = [];
  let currentEdges = [];
  let animFrameId = null;

  // Initialize Canvas Size
  function resizeCanvas() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
  }
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();

  // Load Initial Status
  fetch("/api/status")
    .then(r => r.json())
    .then(data => {
      if (data.device) {
        deviceName.textContent = `${data.device} • Active`;
      }
      if (data.stream_mode) {
        modeSelect.value = data.stream_mode;
        updateModeLabel(data.stream_mode);
      }
      updateKPIs(data.stats);
    })
    .catch(err => console.error("Status fetch error:", err));

  // Load Initial Federated Status
  fetchFederatedStatus();

  // Mode Selection Handler
  modeSelect.addEventListener("change", (e) => {
    const selectedMode = e.target.value;
    updateModeLabel(selectedMode);
    fetch("/api/stream/mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: selectedMode })
    });
  });

  function updateModeLabel(mode) {
    if (mode === "live_system") {
      modeSubText.textContent = "Live Windows host & socket telemetry";
    } else {
      modeSubText.textContent = "NF-ToN-IoT-V2 benchmark replay";
    }
  }

  // Initialize WebSocket Stream
  function initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
    
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log("[+] Connected to SFM-GNN+ Real-Time SOC WebSocket");
      graphOverlay.style.display = "none";
    };

    socket.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "FLOW_EVENT") {
        handleFlowEvent(msg.data, msg.stats);
      }
    };

    socket.onclose = () => {
      console.log("[-] WebSocket closed. Retrying in 2 seconds...");
      setTimeout(initWebSocket, 2000);
    };

    socket.onerror = (err) => {
      console.error("WebSocket error:", err);
    };
  }
  initWebSocket();

  // Handle incoming flow event
  function handleFlowEvent(flow, stats) {
    updateKPIs(stats);
    renderEventCard(flow);
    updateGraph(flow.subgraph);
    
    if (flow.explanation && flow.explanation.top_attributions) {
      renderXAI(flow.predicted_class_name, flow.explanation.top_attributions);
    }

    if (flow.verification && flow.verification.is_confirmed_attack) {
      fetchQuarantines();
    }
  }

  function updateKPIs(stats) {
    if (!stats) return;
    kpiTotal.textContent = stats.total_processed.toLocaleString();
    kpiAttacks.textContent = stats.attacks_confirmed.toLocaleString();
    kpiSuppressed.textContent = stats.fp_suppressed.toLocaleString();
    kpiClean.textContent = stats.normal_allowed.toLocaleString();
    quarantineCount.textContent = (stats.active_quarantines || 0).toString();
  }

  // Render Event in Feed
  function renderEventCard(flow) {
    const meta = flow.metadata || {};
    const verif = flow.verification || {};
    const status = verif.quality_status || "NORMAL";
    const isAttack = verif.is_confirmed_attack;
    const isSuppressed = status === "FALSE_POSITIVE_SUPPRESSED";

    let cardType = "clean";
    let tagClass = "success";
    let tagText = "ALLOW";

    if (isAttack) {
      cardType = "attack";
      tagClass = "danger";
      tagText = "QUARANTINE";
    } else if (isSuppressed) {
      cardType = "suppressed";
      tagClass = "warning";
      tagText = "SUPPRESSED (FP)";
    }

    const card = document.createElement("div");
    card.className = `event-card ${cardType}`;
    
    const timeStr = new Date().toLocaleTimeString();
    const confPercent = (flow.confidence * 100).toFixed(1);
    const procInfo = meta.process_name ? `${meta.process_name} (PID ${meta.cid || 0})` : `Host PID ${meta.cid || 1024}`;

    card.innerHTML = `
      <div class="event-top">
        <span class="event-id">${meta.flow_id || 'FLOW'} • ${timeStr}</span>
        <span class="event-tag ${tagClass}">${tagText}</span>
      </div>
      <div class="event-body">
        <strong>${flow.predicted_class_name.toUpperCase()}</strong>
        <span>Confidence: ${confPercent}%</span>
      </div>
      <div class="event-detail">
        ${procInfo} | ${meta.src_ip || '10.0.0.1'}:${meta.dst_port || 80}
      </div>
    `;

    eventsContainer.insertBefore(card, eventsContainer.firstChild);

    // Keep max 20 cards in feed
    if (eventsContainer.children.length > 20) {
      eventsContainer.removeChild(eventsContainer.lastChild);
    }
  }

  // Render XAI Feature Attribution
  function renderXAI(className, topAttributions) {
    xaiTargetClass.textContent = `Target: ${className.toUpperCase()}`;
    xaiBars.innerHTML = "";

    if (!topAttributions || topAttributions.length === 0) {
      xaiBars.innerHTML = `<div class="empty-state">No prominent features.</div>`;
      return;
    }

    const maxScore = Math.max(...topAttributions.map(a => a.attribution_score), 0.001);

    topAttributions.slice(0, 4).forEach(item => {
      const pct = Math.min((item.attribution_score / maxScore) * 100, 100);
      const barItem = document.createElement("div");
      barItem.className = "xai-item";
      barItem.innerHTML = `
        <div class="xai-meta">
          <span class="xai-name">${item.feature_name}</span>
          <span class="xai-val">score: ${item.attribution_score.toFixed(3)}</span>
        </div>
        <div class="xai-track">
          <div class="xai-fill" style="width: ${pct}%"></div>
        </div>
      `;
      xaiBars.appendChild(barItem);
    });
  }

  // Fetch and Update Quarantine Table
  function fetchQuarantines() {
    fetch("/api/quarantine")
      .then(r => r.json())
      .then(records => {
        quarantineTableBody.innerHTML = "";
        if (!records || records.length === 0) {
          quarantineTableBody.innerHTML = `
            <tr>
              <td colspan="8" class="text-center text-muted" style="padding: 24px;">No active network quarantines. System running clean.</td>
            </tr>
          `;
          return;
        }

        records.forEach(q => {
          const row = document.createElement("tr");
          row.innerHTML = `
            <td>${q.timestamp}</td>
            <td><strong class="text-danger">${q.threat_type}</strong></td>
            <td>${q.confidence}</td>
            <td><code>${q.src_ip}</code></td>
            <td>${q.dst_port}</td>
            <td><code>${q.process_name || 'Host'} (PID ${q.pid})</code></td>
            <td><span class="code-cell">${q.openflow_rule || 'actions=drop'}</span></td>
            <td><button class="btn btn-release" data-id="${q.id}">Release</button></td>
          `;
          quarantineTableBody.appendChild(row);
        });

        // Attach release listeners
        document.querySelectorAll(".btn-release").forEach(btn => {
          btn.addEventListener("click", () => {
            const qId = parseInt(btn.getAttribute("data-id"));
            releaseQuarantine(qId);
          });
        });
      })
      .catch(err => console.error("Quarantine fetch error:", err));
  }

  function releaseQuarantine(qId) {
    fetch("/api/quarantine/release", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quarantine_id: qId })
    })
      .then(r => r.json())
      .then(() => fetchQuarantines())
      .catch(err => console.error("Release error:", err));
  }

  // Stream Toggle
  streamToggleBtn.addEventListener("click", () => {
    fetch("/api/stream/toggle", { method: "POST" })
      .then(r => r.json())
      .then(data => {
        isStreaming = data.stream_active;
        if (isStreaming) {
          streamToggleBtn.innerHTML = `<span class="btn-icon">⏸</span> Pause Stream`;
          streamToggleBtn.classList.remove("paused");
        } else {
          streamToggleBtn.innerHTML = `<span class="btn-icon">▶</span> Resume Stream`;
          streamToggleBtn.classList.add("paused");
        }
      });
  });

  // Manual Attack Injection Buttons
  document.querySelectorAll(".btn-attack, .btn-clean").forEach(btn => {
    btn.addEventListener("click", () => {
      const attackType = btn.getAttribute("data-attack");
      btn.style.opacity = "0.5";
      fetch("/api/inject", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ attack_type: attackType })
      })
        .then(r => r.json())
        .then(() => {
          btn.style.opacity = "1";
        })
        .catch(err => {
          console.error("Injection error:", err);
          btn.style.opacity = "1";
        });
    });
  });

  // Module 3: Federated Learning Functions
  function fetchFederatedStatus() {
    fetch("/api/federated/status")
      .then(r => r.json())
      .then(data => {
        if (data.clients) {
          data.clients.forEach(c => {
            const pkElem = document.getElementById(`pkNode${c.id}`);
            if (pkElem) pkElem.textContent = c.public_key;
          });
        }
        if (data.latest_round) {
          renderFederatedRoundResult(data.latest_round);
        }
      })
      .catch(err => console.error("Fed status error:", err));
  }

  function triggerFederatedRound(poisonClientId = null) {
    const payload = poisonClientId ? { poison_client_id: poisonClientId } : {};
    runFedRoundBtn.disabled = true;
    testPoisonDefenseBtn.disabled = true;

    fetch("/api/federated/round", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(r => r.json())
      .then(result => {
        renderFederatedRoundResult(result);
        runFedRoundBtn.disabled = false;
        testPoisonDefenseBtn.disabled = false;
      })
      .catch(err => {
        console.error("Federated round error:", err);
        runFedRoundBtn.disabled = false;
        testPoisonDefenseBtn.disabled = false;
      });
  }

  function renderFederatedRoundResult(result) {
    if (!result || !result.clients) return;

    secaggRoundText.textContent = `Round ${result.round} Executed`;
    
    result.clients.forEach(c => {
      const normElem = document.getElementById(`normNode${c.client_id}`);
      const zkElem = document.getElementById(`zkNode${c.client_id}`);
      
      if (normElem) normElem.textContent = `||ΔW|| = ${c.gradient_norm}`;
      if (zkElem) {
        if (c.server_action === "ACCEPTED") {
          zkElem.textContent = "ZK Verified: ACCEPTED";
          zkElem.className = "badge";
          zkElem.style.color = "#10b981";
          zkElem.style.borderColor = "rgba(16, 185, 129, 0.4)";
        } else {
          zkElem.textContent = "DROPPED (Poison Flagged)";
          zkElem.className = "badge badge-danger";
        }
      }
    });

    const secagg = result.secagg_summary || {};
    const zkSum = result.zk_verification_summary || {};

    if (zkSum.dropped_malicious > 0) {
      secaggDetailText.innerHTML = `
        <span class="text-danger">🛡️ ZK Verification Defense Triggered:</span> Client update rejected for exceeding displacement threshold (Poisoning Detected!). Remaining ${secagg.num_clients_aggregated || 0} valid client updates aggregated via SecAgg noise cancellation.
      `;
    } else {
      secaggDetailText.innerHTML = `
        <span class="text-success">✓ SecAgg Complete:</span> All ${secagg.num_clients_aggregated || 3} client updates aggregated blindly. Zero-sum pairwise noise cancelled out mathematically: <code style="color:#38bdf8;">Sum(W_i + M_i) = Sum(W_i)</code>. Plaintext exposure: ZERO.
      `;
    }
  }

  runFedRoundBtn.addEventListener("click", () => triggerFederatedRound(null));
  testPoisonDefenseBtn.addEventListener("click", () => triggerFederatedRound(2)); // Poison Client 2

  // Threat Topology Canvas Graph Renderer
  function updateGraph(subgraph) {
    if (!subgraph || !subgraph.nodes) return;

    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2;

    const positions = [
      { x: cx - 140, y: cy - 40 }, // Process
      { x: cx, y: cy - 80 },       // File
      { x: cx, y: cy + 60 },       // Socket
      { x: cx + 150, y: cy + 10 }  // Flow
    ];

    currentNodes = subgraph.nodes.map((n, i) => ({
      ...n,
      x: positions[i % positions.length].x,
      y: positions[i % positions.length].y,
      radius: n.type === "flow" ? 22 : 18
    }));

    currentEdges = subgraph.edges || [];
    renderCanvasGraph();
  }

  let pulseAngle = 0;
  function renderCanvasGraph() {
    if (animFrameId) cancelAnimationFrame(animFrameId);

    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      pulseAngle += 0.05;

      // Draw Edges
      currentEdges.forEach(edge => {
        const srcNode = currentNodes.find(n => n.id === edge.source);
        const dstNode = currentNodes.find(n => n.id === edge.target);
        if (srcNode && dstNode) {
          ctx.beginPath();
          ctx.moveTo(srcNode.x, srcNode.y);
          ctx.lineTo(dstNode.x, dstNode.y);
          ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
          ctx.lineWidth = 2;
          ctx.stroke();

          // Relation label
          const midX = (srcNode.x + dstNode.x) / 2;
          const midY = (srcNode.y + dstNode.y) / 2;
          ctx.fillStyle = "#64748b";
          ctx.font = "10px Inter";
          ctx.fillText(edge.relation, midX, midY - 4);
        }
      });

      // Draw Nodes
      currentNodes.forEach(node => {
        // Attack pulse effect for threat flow node
        if (node.type === "flow" && node.color === "#ef4444") {
          ctx.beginPath();
          const r = node.radius + Math.sin(pulseAngle) * 6 + 4;
          ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(239, 68, 68, 0.4)";
          ctx.lineWidth = 2;
          ctx.stroke();
        }

        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fillStyle = node.color;
        ctx.shadowColor = node.color;
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.shadowBlur = 0;

        // Border
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();

        // Label
        ctx.fillStyle = "#f8fafc";
        ctx.font = "bold 11px Inter";
        ctx.textAlign = "center";
        ctx.fillText(node.label, node.x, node.y + node.radius + 16);
      });

      animFrameId = requestAnimationFrame(draw);
    }
    draw();
  }
});
