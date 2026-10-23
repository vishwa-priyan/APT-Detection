## 1. Project Title, Problem Definition and Understanding

### Project Title

**SFM-GNN+: Secure Federated Multimodal Graph Neural Networks for Robust Advanced Persistent Threat (APT) Detection**

### Problem Definition & Understanding

Advanced Persistent Threats (APTs) are stealthy, continuous cyberattacks targeting enterprise networks. Detecting APTs across modern distributed infrastructure faces three critical challenges:

1. **Privacy vs. Intelligence Dilemma:** Existing threat detection models rely on centralized aggregation of network and host telemetry, exposing sensitive enterprise logs and model weights to data leaks or gradient inversion attacks.
2. **Multi-Relational & Multi-Platform Complexity:** APTs operate across diverse platforms (Windows, Linux, IoT) and exhibit complex structural relations (processes, sockets, file alterations, dynamic library calls). Standard homogeneous intrusion detection systems (IDS) fail to capture these cross-domain graph dependencies.
3. **Evasion, Poisoning, & Explainability Gaps:** Current Federated Learning (FL) models lack resilience against model poisoning or topological graph perturbations, fail to provide real-time explainability under encrypted pipelines, and offer passive alerts rather than active mitigation.

---

## 2. Literature Review (Existing Solutions)

| Solution / Paper | Key Architecture | Primary Limitations |
| --- | --- | --- |
| **XFedHunter** | Hybrid CNN-GRU (NIDS) + E-GraphSAGE (PIDS) with SHAP and Penultimate Quality Check. | Transmits weight updates in cleartext (vulnerable to gradient inversion); static reliance on port features; lack of privacy protection. |
| **Multimodal HGNN** | Heterogeneous Graph Attention Network (HGNN/GAT) + Isolation Forest + Autoencoder. | Fully centralized data collection model (privacy risk); highly reliant on external OSINT lookup APIs; vulnerable to graph topology evasion. |
| **FedHE-Graph** | Self-Supervised Masked Graph Learning (MAGIC) + Hybrid AES/RSA Encryption. | Decrypts weights into plaintext at the server tier (Single Point of Compromise); lacks model explainability (black-box); degrades under round inflation. |

---

## 3. Proposed Methodology

### System Architecture Block Diagram

```
+------------------------------------------------------------------------------------+
|                                SFM-GNN+ ARCHITECTURE                               |
+------------------------------------------------------------------------------------+

  [ LOCAL CLIENT ENTERPRISE NODE ]                   [ CENTRAL AGGREGATOR SERVER ]
  +-------------------------------------+            +------------------------------------+
  | 1. Data Ingestion & Graph Fusion    |            | 1. Blind SecAgg Aggregator         |
  |    - NetFlow & System Audit Logs    |            |    - Receives [W_i + M_i]           |
  |    - Local Sandbox Execution Traces |            |    - Sums updates (Noise cancels out) |
  |                  |                  |            |    - Zero Plaintext Weight Exposure|
  |                  v                  |            +------------------------------------+
  | 2. Local Heterogeneous GAT Training |                              ^
  |    - Masked Pretraining (MAGIC)     |                              |
  |    - Topological Contrastive Loss   | ---- Masked Weights [W_i + M_i] --+
  |                  |                  |                              |
  |                  v                  |                              v
  | 3. Local Verification & XAI         |            +------------------------------------+
  |    - Sub-Graph GradientSHAP         |            | 2. ZK-SNARK Verification Engine    |
  |    - Penultimate Distance Check     | <--- Global W --- Validates Gradient Bounds   |
  |    - ZK-SNARK Proof Generator       |            |   & Loss Convergence Limits        |
  |                  |                  |            +------------------------------------+
  |                  v                  |
  | 4. SDN Micro-Segmentation Engine    |
  |    - OpenFlow Dynamic Rule Trigger  |
  +-------------------------------------+

```

### System Flowchart

```
                 [ Start: Local System Logs & NetFlow Ingestion ]
                                       |
                                       v
               [ Construct Local Heterogeneous Threat Graph G(V, E) ]
                                       |
                                       v
               [ Train Local Heterogeneous GAT + Contrastive Loss ]
                                       |
                                       v
               [ Generate Pairwise Noise Mask M_i & ZK-Proof pi_i ]
                                       |
                                       v
               [ Transmit Masked Weights (W_i + M_i) & Proof pi_i ]
                                       |
                                       v
            /---------------------------------------------------------\
           / Is ZK-Proof pi_i Valid & Gradient Bounds Compliant?       \
           \---------------------------------------------------------/
                             |                           |
                        (Yes)|                           |(No)
                             v                           v
              [ Aggregate via SecAgg ]         [ Reject Update & Flag Client ]
                             |
                             v
               [ Broadcast Global Weights W ]
                             |
                             v
           /-----------------------------------------------------------\
          / Alert Triggered on Local Node During Forward Pass?         \
          \-----------------------------------------------------------/
                             |                           |
                        (Yes)|                           |(No)
                             v                           v
              [ Perform Penultimate Verification ]   [ Continue Monitoring ]
                             |
                             v
                  /-------------------------\
                 / Confirmed True Positive?  \
                 \-------------------------/
                       |              |
                  (Yes)|              |(No / FP)
                       v              v
         [ Execute SDN Flow Rule  ]  [ Log False Alarm ]
         [ & Generate Local XAI   ]

```

### Core Algorithms

#### Algorithm 1: Secure Federated Aggregation with ZK Verification (Server & Client Tier)

```text
1: Input: Client weight updates W_i, pairwise noise masks M_i, ZK-proofs pi_i, gradient bound delta
2: Procedure Client_Update_And_Mask(Client i):
3:    Compute local weight update W_i^t via HGNN training
4:    Generate pairwise noise masks M_i using Diffie-Hellman seed exchange s.t. Sum(M_i) = 0
5:    Generate ZK-SNARK proof pi_i asserting ||W_i^t - W_global^{t-1}|| <= delta
6:    Transmit (W_i^t + M_i) and pi_i to Aggregator Server
7: End Procedure

8: Procedure Server_Aggregate(Masked Updates, Proofs):
9:    For each client i do:
10:       If Verify_ZK_Proof(pi_i) == True and ||W_i_masked|| <= Threshold then
11:          Accept masked update
12:       Else: Drop client update from communication round
13:    End For
14:    Compute W_global^t = Sum(W_i^t + M_i) = Sum(W_i^t)  // Noise terms cancel out
15:    Broadcast W_global^t to all active client nodes
16: End Procedure

```

#### Algorithm 2: Local Penultimate Decision Quality & SDN Mitigation

```text
1: Input: Test sample x, trained local model f, penultimate layer activation f^(-1)(x)
2: Function Verify_And_Mitigate(x):
3:    Extract penultimate representation v_x = f^(-1)(x)
4:    Compute cluster distance d_k = Euclidean_Distance(v_x, C_k) for k in {TP, TN, FP, FN}
5:    Assign verified class c = argmin(d_k)
6:    If c == True_Positive then
7:       Generate sub-graph attribution map via GradientSHAP
8:       Formulate SDN OpenFlow Rule: Drop flow for Target_IP/MAC and Isolate Process PID
9:       Send REST command to SDN Controller
10:   Else If c == False_Positive then
11:      Suppress alert and re-fit penultimate background baseline
12:   End If
13: End Function

```

### Techniques Used for Implementation

* **Heterogeneous Graph Neural Networks (PyTorch Geometric):** Multi-relational GAT layers with type-specific feature projection.
* **Blind Secure Aggregation (SecAgg):** Pairwise Diffie-Hellman mask generation to achieve mathematical privacy without server decryption.
* **Zero-Knowledge Proofs (pySNARK / ZoKrates):** Circuit generation verifying local gradient norm bounds.
* **Robust Graph Pretraining:** Topological Contrastive Learning (TCL) with InfoNCE loss to build resilience against edge perturbations.
* **Explainability & Verification:** Local GradientSHAP + Penultimate distance clustering.

---

## 4. Hardware and Software Specifications

### Hardware Specifications

* **Processor:** Intel Core i7 / i9 (12th Gen or higher) or AMD Ryzen 7/9
* **GPU (Crucial for GNN Training):** NVIDIA RTX 3060 / 4070 (Minimum 8GB VRAM) or Google Colab T4/A100 GPU
* **System Memory (RAM):** 16 GB DDR4/DDR5 (32 GB recommended)
* **Storage:** 512 GB NVMe Solid State Drive (SSD)

### Software Specifications

* **Operating System:** Ubuntu 22.04 LTS / Windows 11 WSL2
* **Programming Language:** Python 3.10+
* **Deep Learning Framework:** PyTorch 2.x, PyTorch Geometric (PyG)
* **Federated Learning Framework:** FedML / OpenMined PySyft
* **Cryptographic & ZK Libraries:** PyCryptodome, ZoKrates / pySNARK
* **SDN Controller Integration:** Ryu Controller / ONOS via OpenFlow v1.3
* **XAI & Analytics:** SHAP, Scikit-learn, NetworkX, Matplotlib

---

## 5. Module Description

* **Module 1: Multimodal Data Ingestion & Graph Construction Engine**
Processes raw NetFlow PCAPs, system audit logs (FreeBSD/Linux), and local dynamic sandbox traces (Cuckoo) to construct multi-relational heterogeneous threat graphs $\mathcal{G} = (\mathcal{V}, \mathcal{E})$.
* **Module 2: Self-Supervised Heterogeneous GNN Module**
Implements multi-head GAT attention layers combined with Masked Representation Learning (MAGIC) and Topological Contrastive Learning to learn feature representations robust to graph perturbations.
* **Module 3: Blind Secure Aggregation & ZK-Verification Pipeline**
Handles client-side zero-sum noise masking (SecAgg) and ZK-SNARK proof generation. Aggregates updates at the server without decrypting plaintext weights, while dropping poisoned updates.
* **Module 4: Decoupled Local Explainer & Penultimate Verification**
Runs locally on client nodes to extract 64-dimensional penultimate vectors, compute $k$-NN cluster reliability distances (TP, TN, FP, FN), and output GradientSHAP sub-graph plots for security teams.
* **Module 5: Automated SDN Remediation & Self-Healing Engine**
Translates verified TP detections into OpenFlow micro-segmentation commands, directing the SDN controller to isolate compromised host nodes and terminate malicious processes automatically.

---

## 6. Module Responsibilities & Timeline Chart

### Member Responsibilities

```
+-----------------------------------------------------------------------------------------+
|                                  TEAM RESPONSIBILITIES                                  |
+-----------------------------------------------------------------------------------------+
|  Member 1 (GNN Architecture & Graph Pipelines):                                        |
|  - Module 1 (Multimodal Data Ingestion & Graph Construction Engine)                     |
|  - Module 2 (Self-Supervised Heterogeneous GNN & Contrastive Learning)                  |
|                                                                                         |
|  Member 2 (Privacy, Cryptography & FL Pipeline):                                        |
|  - Module 3 (Blind SecAgg Engine & ZK-SNARK Proof Verification)                         |
|  - Federated Server Coordination & Communication Setup                                  |
|                                                                                         |
|  Member 3 (Explainability, Verification & SDN Orchestration):                           |
|  - Module 4 (Decoupled GradientSHAP & Penultimate Distance Verification)                |
|  - Module 5 (SDN Controller Integration & OpenFlow Rule Automation)                     |
+-----------------------------------------------------------------------------------------+

```

### Timeline Chart (Gantt Chart Representation)

```
+------------------------------------------------------------------------------------+
| Month / Week  | Task / Activity                                | Assigned Member   |
+------------------------------------------------------------------------------------+
| W1 - W2       | Dataset Preprocessing (DARPA/TON-IoT) & Graphs | Member 1          |
| W3 - W4       | Local HGNN & Contrastive Model Development     | Member 1          |
| W5 - W6       | SecAgg Protocol & ZK-SNARK Proof Implementation| Member 2          |
| W7 - W8       | Federated Aggregation Engine & Server Setup    | Member 2          |
| W9 - W10      | Local Penultimate Check & GradientSHAP XAI     | Member 3          |
| W11 - W12     | SDN OpenFlow Integration & Testing             | Member 3          |
| W13 - W14     | System Integration, Benchmarking & Report     | All Members       |
+------------------------------------------------------------------------------------+

```
---

## 8. Conclusion and Future Enhancement

### Conclusion

**SFM-GNN+** provides a complete solution for distributed APT threat hunting. By combining Blind Secure Aggregation (SecAgg) with Zero-Knowledge verification, the framework addresses the privacy vulnerabilities present in existing federated architectures. Extending GNN models to heterogeneous multi-relational graph schemas captures complex attack semantics across disparate operating systems. Furthermore, integrating local penultimate vector verification with automated SDN controllers establishes a closed-loop framework capable of mitigating verified threats in real time.

### Future Enhancement

1. **Fully Homomorphic Encryption (FHE) Acceleration:** Transition from SecAgg to GPU-accelerated FHE schemes to support non-interactive aggregation across asynchronous client connections.
2. **LLM-Powered Forensic Summarization:** Connect the GradientSHAP sub-graph output to a fine-tuned local Large Language Model (LLM) to automatically draft incident reports for security operations center (SOC) analysts.
3. **Cross-Silo Autonomous Federated Learning:** Extend the pipeline to support cross-organizational threat intelligence sharing without requiring a shared central server.

---

## 9. References

1. Huynh Thai Thi, et al., "XFedHunter: An Explainable Federated Learning Framework for Advanced Persistent Threat Detection in SDN," *IEEE Transactions*, 2024.
2. Premanand Ghadekar, et al., "Multimodal Advanced Persistent Threat Detection and Attribution Using Heterogeneous Graph Neural Network and Explainable AI," *Springer/IEEE*, 2024.
3. Athmane Ayoub Mansour Bahar, et al., "FedHE-Graph: Federated Learning with Hybrid Encryption on Graph Neural Networks for Advanced Persistent Threat Detection," *Elsevier Computer Networks*, 2024.
4. Bonawitz, K., et al., "Practical Secure Aggregation for Privacy-Preserving Machine Learning," *ACM CCS*, 2017.
5. Wang, Y., et al., "MAGIC: Masked Graph Representation Learning for APT Detection," *IEEE Symposium on Security and Privacy (S&P)*, 2023.