import time
import os
from typing import Dict, List, Any, Optional
import numpy as np
import torch

from src.config import (
    SAVED_MODELS_DIR,
    ATTACK_CLASSES,
    CLASS_TO_IDX,
    NETFLOW_FEATURE_COLS,
)
from src.data.netflow_loader import NetFlowLoader
from src.data.graph_builder import ThreatGraphBuilder, HeteroThreatGraph
from src.models.hetero_gat import SFMGNNPlusModel
from src.models.penultimate_verifier import PenultimateVerifier
from src.explainability.xai_explainer import ThreatGraphExplainer
from src.remediation.sdn_controller import SDNOpenFlowRemediator

class RealtimeDetectionEngine:
    """
    Real-Time Streaming Threat Detection Engine:
    Processes streaming network flows and host events in real time,
    maintains graph windows, executes HeteroGAT inference, penultimate verification,
    and automated SDN remediation.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cpu")
        self.model = SFMGNNPlusModel().to(self.device)
        self.netflow_loader = NetFlowLoader()
        self.penultimate_verifier = PenultimateVerifier()
        self.graph_builder = ThreatGraphBuilder()
        self.remediator = SDNOpenFlowRemediator()
        self.explainer = None
        self.is_ready = False

        if model_path or (SAVED_MODELS_DIR / "sfm_gnn_model.pt").exists():
            self.load_model(model_path)

    def load_model(self, model_path: Optional[str] = None):
        path = model_path or str(SAVED_MODELS_DIR / "sfm_gnn_model.pt")
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()

        self.netflow_loader.load_scaler()
        self.penultimate_verifier.load()
        self.graph_builder.load_host_context(max_records=10000)
        self.explainer = ThreatGraphExplainer(self.model)
        self.is_ready = True

    def process_flow_batch(
        self,
        raw_flows: np.ndarray,
        flow_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Processes a live batch or single streaming flow through the end-to-end detection pipeline.
        raw_flows: [N, 41] raw flow features
        """
        if not self.is_ready:
            raise RuntimeError("Detection engine is not loaded. Call load_model() first.")

        start_time = time.perf_counter()
        
        # 1. Normalize features
        scaled_flows = self.netflow_loader.transform_features(raw_flows)
        num_flows = len(scaled_flows)
        dummy_labels = np.zeros(num_flows, dtype=np.int64)

        # 2. Build or update graph window
        graph = self.graph_builder.construct_fused_graph(
            flow_features=scaled_flows,
            flow_labels=dummy_labels,
            max_host_nodes=100
        )

        # Move to device
        node_feats = {k: v.to(self.device) for k, v in graph.node_features.items()}
        edge_indices = {k: v.to(self.device) for k, v in graph.edge_indices.items()}

        # 3. Model Forward Pass
        with torch.no_grad():
            logits, penultimate = self.model(node_feats, edge_indices, return_penultimate=True)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            penultimate_np = penultimate.cpu().numpy()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        avg_latency_per_flow_ms = elapsed_ms / max(num_flows, 1)

        results = []
        for i in range(num_flows):
            pred_class = int(np.argmax(probs[i]))
            conf = float(probs[i][pred_class])
            pen_vec = penultimate_np[i]

            # 4. Penultimate Verification
            verif = self.penultimate_verifier.verify_sample(
                penultimate_vec=pen_vec,
                predicted_class=pred_class,
                confidence=conf
            )

            meta = flow_metadata[i] if (flow_metadata and i < len(flow_metadata)) else {}
            src_ip = meta.get("src_ip", f"10.0.0.{100 + i % 150}")
            dst_port = int(raw_flows[i][1]) if len(raw_flows[i]) > 1 else 80

            # 5. Remediation & Explanation if confirmed attack
            mitigation = None
            explanation = None
            if verif.get("is_confirmed_attack"):
                mitigation = self.remediator.trigger_mitigation(
                    detection_result=verif,
                    host_process_cid=meta.get("cid", 1000 + i),
                    src_ip=src_ip,
                    dst_port=dst_port
                )
                explanation = self.explainer.explain_flow(
                    node_feats=node_feats,
                    edge_indices=edge_indices,
                    flow_idx=i,
                    target_class=pred_class,
                    top_k=4
                )

            res = {
                "flow_index": i,
                "predicted_class_name": ATTACK_CLASSES[pred_class],
                "confidence": conf,
                "verification": verif,
                "mitigation": mitigation,
                "explanation": explanation,
                "latency_ms": avg_latency_per_flow_ms,
                "timestamp": time.time()
            }
            results.append(res)

        return results
