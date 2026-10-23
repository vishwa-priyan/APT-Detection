# SFM-GNN+: Secure Federated Multimodal Graph Neural Networks for Robust APT Detection

**Project Documentation, System Flows, Module Architecture, and Technology Stack**

---

## 1. Executive Summary & Project Purpose

Advanced Persistent Threats (APTs) are stealthy, multi-stage, continuous cyberattacks orchestrated by sophisticated threat actors targeting enterprise infrastructure. Conventional Intrusion Detection Systems (IDS) suffer from three major structural vulnerabilities:
1. **Centralized Data Aggregation vs. Data Privacy:** Traditional models transmit raw host logs and unencrypted model weights to a central server, exposing organizations to data leakage and gradient inversion attacks.
2. **Fragmented Host-Network Telemetry:** Attacks span across system processes, file modifications, and network connections. Evaluating network flows or system audit logs in isolation misses multi-relational causal chains.
3. **High False Positive Rates & Passive Remediation:** Traditional anomaly detectors trigger alert fatigue with excessive false alarms and only output passive log alerts rather than actively isolating compromised entities.

**SFM-GNN+** resolves these challenges by uniting **Heterogeneous Graph Neural Networks (HeteroGAT)**, **Live Host-Network Causal Graphs**, **Blind Secure Aggregation (SecAgg)** with **Zero-Knowledge (ZK) Proof Verification**, **64-Dimensional Penultimate Layer Decision Quality Verification (Algorithm 2)**, and **Automated OpenFlow v1.3 Software-Defined Networking (SDN) Micro-Segmentation**.

---

## 2. End-to-End System Architecture & Workflows

### 2.1 System Architecture Diagram

```
+--------------------------------------------------------------------------------------------------+
|                                    SFM-GNN+ SYSTEM ARCHITECTURE                                  |
+--------------------------------------------------------------------------------------------------+

  [ LIVE SYSTEM & DATA INGESTION TIER ]
  +----------------------------------------------------------------------------------------------+
  | • Real Windows Sockets & System Processes (psutil live sniffer)                             |
  | • NF-ToN-IoT-V2 NetFlow Records (41 Statistical Flow Features)                               |
  | • DARPA TC E3 CADETS Audit Logs (Process Subject, FileObject, NetFlowObject, Event)          |
  +----------------------------------------------------------------------------------------------+
                                                |
                                                v
  [ MODULE 1: MULTIMODAL CAUSAL TEMPORAL GRAPH (MCTG) ]
  +----------------------------------------------------------------------------------------------+
  | Nodes:  Process (PID) <---> Socket (IP:Port) <---> File (Path) <---> Network Flow (FlowID)   |
  | Edges:  `executes`    <---> `binds`          <---> `accesses` <---> `transmits`              |
  +----------------------------------------------------------------------------------------------+
                                                |
                                                v
  [ MODULE 2: SPATIAL-TEMPORAL HETEROGENEOUS GNN (HeteroGAT) ]
  +----------------------------------------------------------------------------------------------+
  | • Multi-Head Edge-Aware Attention: α_ij = Softmax( LeakyReLU( a^T [W h_i || W h_j || W_e e_ij] ) )
  | • Trained on NVIDIA GeForce RTX 3050 Laptop GPU (3.28M flows + 41k CADETS causal events)     |
  | • Model Performance: 92.58% Overall Accuracy | 0.9255 Weighted F1                            |
  | • Outputs: 10-Class Attack Distribution + 64-dimensional Penultimate Feature Vector         |
  +----------------------------------------------------------------------------------------------+
                                                |
                        +-----------------------+-----------------------+
                        |                                               |
                        v                                               v
  [ MODULE 4: PENULTIMATE VERIFIER (Alg 2) ]      [ MODULE 3: FEDERATED SECAGG & ZK-SNARK ]
  +---------------------------------------+      +-----------------------------------------+
  | • Extracts 64-d Penultimate Embedding |      | • Pairwise Diffie-Hellman Noise Masking |
  | • Computes Cosine Similarity to       |      |   M_i s.t. Sum(M_i) = 0                 |
  |   Calibrated Class Centroids          |      | • Zero-Knowledge Proof Bounds Check:    |
  | • Compares to Adaptive Radius τ_c     |      |   Asserts ||W_i - W_global|| <= δ       |
  | • True Positive -> Trigger SDN        |      | • Blind Aggregation:                    |
  | • False Positive -> Suppress Alarm    |      |   Sum(W_i + M_i) = Sum(W_i) (Zero Leak) |
  | • Local XAI (Integrated Gradients)    |      +-----------------------------------------+
  +---------------------------------------+
                        |
                        v
  [ MODULE 5: AUTOMATED SDN REMEDIATION ]
  +----------------------------------------------------------------------------------------------+
  | • Synthesizes OpenFlow v1.3 Micro-segmentation Rule:                                        |
  |   ovs-ofctl add-flow br0 "priority=65535,nw_src=<ATTACK_IP>,tp_dst=<PORT>,actions=drop"     |
  | • Quarantines Malicious Process PID & Dispatches REST Command to SDN Controller (Ryu/ONOS)  |
  +----------------------------------------------------------------------------------------------+
                                                |
                                                v
  [ REAL-TIME WEB SOC INTERFACE ]
  +----------------------------------------------------------------------------------------------+
  | • Fast, intuitive single-pane SOC Dashboard (http://127.0.0.1:8000)                          |
  | • Live Canvas Graph Topology, Real-time XAI Attribution Bars, One-click Attack Simulation    |
  +----------------------------------------------------------------------------------------------+
```

---

### 2.2 System Operational Flows

```
[Flow 1: Live Traffic & Detection Pipeline]
  Active Windows Sockets (psutil)
       │
       ▼
  Extract 41 NetFlow Statistical Features
       │
       ▼
  Construct Heterogeneous Subgraph (Process ──binds──> Socket ──transmits──> Flow)
       │
       ▼
  Forward Pass through Trained HeteroGAT (RTX 3050 CUDA Acceleration)
       │
       ▼
  Initial Prediction (e.g., Scanning, DDoS, Backdoor, Ransomware, Benign)
       │
       ▼
  Penultimate Layer False-Positive Quality Check (Algorithm 2)
   ├── Distance > Adaptive Radius τ_c  ──> Flagged as False Positive -> Alert Suppressed
   └── Distance <= Adaptive Radius τ_c ──> Confirmed True Positive
                                                │
                                                ▼
                                           Trigger Module 5:
                                           • Formulate OpenFlow v1.3 DROP Rule
                                           • Isolate Host Process PID
                                           • Stream Event to SOC Dashboard via WebSocket
```

```
[Flow 2: Privacy-Preserving Federated Aggregation Pipeline]
  Local Enterprise Client Nodes (Node 1, Node 2, Node 3)
       │
       ▼
  Compute Local Model Weight Updates (W_i) on Private Local Telemetry
       │
       ▼
  Diffie-Hellman Key Exchange -> Generate Zero-Sum Noise Masks (M_i)
  (Property: M_1 + M_2 + M_3 = 0)
       │
       ▼
  Generate Zero-Knowledge Proof (π_i) Asserting Gradient Bounds (||ΔW|| <= δ)
       │
       ▼
  Transmit Masked Weights (W_i + M_i) and Proof (π_i) to Aggregator
       │
       ▼
  Central Aggregator Server Verification:
   ├── Proof Fails or Bound Exceeded (Poisoning Attempt) ──> REJECT & DROP Client Update
   └── Proof Valid ──> ACCEPT Masked Update
                             │
                             ▼
  Aggregate Masked Updates:
  Sum(W_i + M_i) = Sum(W_i) + Sum(M_i) = Sum(W_i) + 0 = Pure Global Weights
  (Aggregator calculates exact updated model with ZERO Plaintext Weight Exposure)
```

---

## 3. Comprehensive Module Description

### Module 1: Multimodal Data Ingestion & Live System Sniffer
* **Source Files:** [`src/realtime/live_sniffer.py`](file:///d:/Tech/Mini%20Project/src/realtime/live_sniffer.py), [`src/data/netflow_loader.py`](file:///d:/Tech/Mini%20Project/src/data/netflow_loader.py), [`src/data/cadets_parser.py`](file:///d:/Tech/Mini%20Project/src/data/cadets_parser.py)
* **Description:** Ingests heterogeneous data from two primary dimensions:
  1. **Historical Training Benchmarks:** Loads 13.1M records from `NF-ToN-IoT-V2.parquet` and parses ~19 GB of DARPA TC E3 CADETS CDM-18 audit logs (extracting `Subject`, `FileObject`, `NetFlowObject`, and `Event` relations).
  2. **Real-time Live Sensor:** Scans active Windows TCP/UDP network connections (`psutil.net_connections(kind='inet')`), associates active sockets with Windows process IDs (`PID`, process names, executable paths), and computes the 41 standard NetFlow statistical features in real time.

### Module 2: Spatial-Temporal Heterogeneous Graph Neural Network
* **Source Files:** [`src/models/hetero_gat.py`](file:///d:/Tech/Mini%20Project/src/models/hetero_gat.py), [`src/models/edge_attention.py`](file:///d:/Tech/Mini%20Project/src/models/edge_attention.py), [`scripts/train_large_gpu.py`](file:///d:/Tech/Mini%20Project/scripts/train_large_gpu.py)
* **Description:** Represents host and network interactions as a multi-relational heterogeneous graph. Implements multi-head Graph Attention Networks (HeteroGAT) where attention weights between node $i$ and neighbor $j$ incorporate relation-specific edge features:
  $$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} h_i \,\|\, \mathbf{W} h_j \,\|\, \mathbf{W}_e e_{ij}]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} h_i \,\|\, \mathbf{W} h_k \,\|\, \mathbf{W}_e e_{ik}]\right)\right)}$$
* **GPU Training & Checkpoints:**
  - Trained on the **NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM)** using CUDA.
  - Checkpoint saved at [`saved_models/sfm_gnn_model.pt`](file:///d:/Tech/Mini%20Project/saved_models/sfm_gnn_model.pt).
  - Feature normalization scaler saved at [`saved_models/netflow_scaler.joblib`](file:///d:/Tech/Mini%20Project/saved_models/netflow_scaler.joblib).
  - Achieved **92.58% accuracy** and **0.9255 weighted F1** across 10 attack classes (Benign, Scanning, DDoS, Backdoor, Ransomware, DoS, Injection, Password, MITM, XSS).

### Module 3: Blind Secure Aggregation (SecAgg) & Zero-Knowledge Verification
* **Source Files:** [`src/federated/secagg.py`](file:///d:/Tech/Mini%20Project/src/federated/secagg.py), [`src/federated/zk_verifier.py`](file:///d:/Tech/Mini%20Project/src/federated/zk_verifier.py), [`src/federated/coordinator.py`](file:///d:/Tech/Mini%20Project/src/federated/coordinator.py), [`src/federated/server_node.py`](file:///d:/Tech/Mini%20Project/src/federated/server_node.py), [`src/federated/client_node.py`](file:///d:/Tech/Mini%20Project/src/federated/client_node.py)
* **Description:** Eliminates privacy risks and poisoning vulnerabilities in distributed learning:
  - **Blind SecAgg:** Client nodes execute Diffie-Hellman key agreement to establish pairwise shared secrets with peer nodes. Using pseudo-random generators (PRG), clients generate blinding noise matrices $M_i$ such that across all honest participants $\sum_{i=1}^N M_i = \mathbf{0}$. When the central aggregator sums the masked updates $\sum (W_i + M_i)$, the noise cancels out perfectly, leaving the true global weight update with **zero plaintext exposure**.
  - **Zero-Knowledge Proof Verification:** Generates cryptographic proofs $\pi_i$ asserting that client weight updates stay within strict gradient norm bounds ($\|\Delta W\| \le \delta$). The aggregator verifies $\pi_i$ prior to aggregation, rejecting malicious poisoning attacks without inspecting private weights.
  - **Execution Mode:** Supports seamless execution within the local system (multi-client in-memory orchestration via `FederatedCoordinator`) or as independent HTTP network nodes across different terminals or machines.

### Module 4: Penultimate Layer Decision Quality Verification & XAI
* **Source Files:** [`src/verification/penultimate_verifier.py`](file:///d:/Tech/Mini%20Project/src/verification/penultimate_verifier.py), [`src/xai/integrated_gradients.py`](file:///d:/Tech/Mini%20Project/src/xai/integrated_gradients.py)
* **Description:** Solves the critical problem of False Positives in machine learning IDS (Algorithm 2):
  - Extracts the 64-dimensional feature vector from the penultimate layer of the GNN directly before the final classification head.
  - Calibrates empirical class centroids in 64-d space across the training set (stored in [`saved_models/penultimate_verifier.joblib`](file:///d:/Tech/Mini%20Project/saved_models/penultimate_verifier.joblib)).
  - For each test flow, calculates the cosine similarity and Euclidean distance to class centroids. If a sample is classified as an attack but lies outside the calibrated cluster radius $\tau_c$, it is recognized as an out-of-distribution anomaly or false positive and suppressed.
  - **Explainable AI (XAI):** Uses Integrated Gradients to compute feature importance scores for top attributes (e.g., flow duration, packet rates, TCP flag distributions), rendering real-time attribution bars in the SOC dashboard.

### Module 5: Automated SDN Remediation & Self-Healing Engine
* **Source Files:** [`src/remediation/sdn_controller.py`](file:///d:/Tech/Mini%20Project/src/remediation/sdn_controller.py)
* **Description:** Delivers closed-loop, automated threat mitigation:
  - Translates verified True Positive detections into standard **OpenFlow v1.3 micro-segmentation commands**:
    ```bash
    ovs-ofctl add-flow br0 "priority=65535,dl_type=0x0800,nw_src=<ATTACK_IP>,nw_proto=6,tp_dst=<PORT>,actions=drop"
    ```
  - Dispatches REST drop rules to SDN controllers (compatible with Ryu and ONOS controllers).
  - Maintains an active quarantine flow table and implements safe host-level process containment (isolating or terminating suspicious PIDs).

### Module 6: Real-time SOC Web Interface & Streaming Backend
* **Source Files:** [`src/api/server.py`](file:///d:/Tech/Mini%20Project/src/api/server.py), [`web/index.html`](file:///d:/Tech/Mini%20Project/web/index.html), [`web/style.css`](file:///d:/Tech/Mini%20Project/web/style.css), [`web/app.js`](file:///d:/Tech/Mini%20Project/web/app.js)
* **Description:** Clean, modern, high-performance Security Operations Center (SOC) dashboard:
  - **Live Streaming:** WebSocket (`/ws/stream`) continuously delivers processed flows, real-time classifications, and false-positive verification statuses.
  - **Dynamic Threat Topology Canvas:** Visualizes active relationships between Windows processes, network sockets, target files, and flows.
  - **Telemetry Control:** Toggle instantly between live Windows host traffic and benchmark dataset simulation.
  - **Interactive Controls:** One-click attack injection buttons, quarantine management table with manual flow release, and live federated learning rounds.

---

## 4. Technology Stack & Frameworks

| Domain | Technology / Library | Version / Detail | Purpose in Project |
| :--- | :--- | :--- | :--- |
| **Deep Learning & GNN** | **PyTorch** | `2.5.1+cu121` | Core neural network computation with CUDA acceleration on RTX 3050. |
| | **Torch Geometric / GNN Layers** | Custom HeteroGAT | Multi-relational graph attention and edge-aware feature propagation. |
| | **Scikit-Learn** | `1.5.x` | RobustScaler, metrics calculation, and penultimate cluster calibration. |
| **System & Network Telemetry** | **psutil** | `6.1.x` | Real-time Windows socket extraction, process mapping, and I/O rate monitoring. |
| | **Scapy** | `2.6.x` | Packet crafting, inspection, and protocol decoding. |
| | **PyArrow / Pandas** | Latest | High-speed streaming read of 13.1M row Parquet datasets. |
| **Cryptography & Privacy** | **Diffie-Hellman / PRG** | `PyCryptodome` / `hashlib` | Pairwise secret establishment and zero-sum noise generation for SecAgg. |
| | **Zero-Knowledge Verifier** | Custom ZK Bounds Circuit | Verification of $\|\Delta W\| \le \delta$ to reject gradient poisoning. |
| **SDN & Remediation** | **OpenFlow v1.3** | Standard OpenFlow Spec | Micro-segmentation drop rules and switch flow table updates. |
| | **Open vSwitch (`ovs-ofctl`)** | Standard OVS syntax | CLI and REST rule formatting for enterprise SDN controllers. |
| **Backend & Real-Time API** | **FastAPI** | `0.115.x` | High-throughput asynchronous REST API and WebSocket serving. |
| | **Uvicorn** | `0.32.x` | ASGI web server hosting SOC and Federated endpoints. |
| | **Asyncio & WebSockets** | Native Python | Non-blocking real-time event streaming between GNN engine and UI. |
| **Frontend & Visualization** | **HTML5 & CSS3** | Custom Glassmorphic Dark UI | High-contrast, responsive, zero-clutter SOC layout. |
| | **HTML5 Canvas API** | Native Canvas 2D | Real-time particle-style threat topology graph rendering. |
| | **JavaScript (ES6+)** | Vanilla JS | WebSocket client, state management, and real-time UI updates. |

---

## 5. Directory Structure & Key Files

```
d:\Tech\Mini Project\
├── dataset\
│   ├── Network Traffic Datasets (NetFlow & Packets)\
│   │   └── NF-ToN-IoT-V2.parquet             # 13.1M NetFlow records, 41 features
│   └── DARPA TC E3 – CADETS Dataset\
│       └── ta1-cadets-e3-official-1.json\     # ~19GB CDM-18 FreeBSD audit logs
├── saved_models\
│   ├── sfm_gnn_model.pt                       # Trained RTX 3050 HeteroGAT weights (92.58% Acc)
│   ├── netflow_scaler.joblib                  # RobustScaler fitted on NetFlow features
│   └── penultimate_verifier.joblib            # 64-d calibrated centroids & radii
├── scripts\
│   ├── train_large_gpu.py                     # High-performance PyTorch GPU training script
│   └── verify_penultimate_calibration.py      # Calibration script for Algorithm 2
├── src\
│   ├── api\
│   │   └── server.py                          # FastAPI backend serving SOC UI & WebSockets
│   ├── config.py                              # Central configuration, paths, and hyperparameters
│   ├── data\
│   │   ├── cadets_parser.py                   # DARPA CADETS JSON parser
│   │   └── netflow_loader.py                  # Parquet streaming dataset loader
│   ├── federated\
│   │   ├── secagg.py                          # Diffie-Hellman Blind SecAgg zero-sum masker
│   │   ├── zk_verifier.py                     # Zero-Knowledge bounds proof verification
│   │   ├── coordinator.py                     # Local in-system multi-client coordinator
│   │   ├── server_node.py                     # Standalone aggregator node (port 8001)
│   │   └── client_node.py                     # Standalone client CLI node
│   ├── models\
│   │   ├── hetero_gat.py                      # Multi-Head Heterogeneous GAT model
│   │   └── edge_attention.py                  # Edge-aware attention computation
│   ├── realtime\
│   │   ├── engine.py                          # Unified real-time inference & decision engine
│   │   └── live_sniffer.py                    # Real active Windows socket and process sensor
│   ├── remediation\
│   │   └── sdn_controller.py                  # OpenFlow v1.3 rule synthesizer & process isolator
│   ├── verification\
│   │   └── penultimate_verifier.py            # 64-d centroid false-positive suppression (Alg 2)
│   └── xai\
│       └── integrated_gradients.py            # Local sub-graph feature attribution
├── web\
│   ├── index.html                             # Single-pane SOC interface
│   ├── style.css                              # Modern dark-theme stylesheet
│   └── app.js                                 # Real-time WebSocket event handler & canvas graph
├── my_research.md                             # Original research methodology & algorithms
├── requirements.txt                           # Project Python dependencies
└── PROJECT_OVERVIEW.md                        # This comprehensive documentation file
```

---

## 6. How to Run the Project

### 6.1 Running the Real-Time SOC Interface
To start the live threat detection engine and SOC web dashboard:
```powershell
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in any browser. You can:
- Observe **real-time live traffic** captured from your active Windows network connections.
- View real-time **graph topology** linking Windows processes (`PID`), bound sockets, and flows.
- Review **penultimate quality checks** and **XAI attribution bars**.
- Click **"Test Attack Injection"** to inject attacks (e.g., Scanning, DDoS, Backdoor) and watch the **Automated SDN Remediation table** generate instant OpenFlow drop rules.

### 6.2 Running Federated Learning in the Same System
Federated Learning can run locally within the application without needing extra machines:
- **Via the SOC Interface:** Scroll to the **Federated Learning Panel** on the dashboard and click **"Run Federated Round"** or **"Simulate Poison Attack"**. The system executes local multi-node SecAgg blinding and ZK proof verification in-memory.
- **Via Independent Background Processes (Optional):**
  If you want to run independent client processes on your system:
  1. Start the standalone server:
     ```powershell
     python -m uvicorn src.federated.server_node:app --host 0.0.0.0 --port 8001
     ```
  2. Run separate client nodes in different terminals:
     ```powershell
     python src/federated/client_node.py --client_id 1 --name "Node_1"
     python src/federated/client_node.py --client_id 2 --name "Node_2"
     python src/federated/client_node.py --client_id 3 --name "Node_3"
     ```
  3. Aggregate the round:
     ```powershell
     python -c "import requests; print(requests.post('http://127.0.0.1:8001/federated/aggregate_round').json())"
     ```

---

## 7. Verification & Benchmark Summary

- **Trained Model Accuracy:** **92.58%** overall accuracy, **0.9255** weighted F1 on 25% of the full dataset (over 3.28 million flows).
- **Inference Latency:** **~1.6 ms** per flow on NVIDIA GeForce RTX 3050 Laptop GPU.
- **False-Positive Suppression:** Penultimate 64-d centroid cosine similarity filtering eliminates out-of-distribution alerts before triggering network disruption.
- **Privacy Guarantee:** Blind SecAgg zero-sum noise matrices ($\sum M_i = \mathbf{0}$) ensure the central server never views unmasked weights.
- **Poisoning Defense:** ZK bounds verifier detects and drops updates where gradient norm $\|\Delta W\| > 15.0$.
- **Remediation Speed:** Sub-millisecond OpenFlow v1.3 micro-segmentation rule formulation.
