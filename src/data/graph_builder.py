import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch

from src.config import (
    PROCESS_FEAT_DIM,
    FILE_FEAT_DIM,
    SOCKET_FEAT_DIM,
    FLOW_FEAT_DIM,
    CADETS_DIR,
)
from src.data.cadets_parser import CADETSParser

class HeteroThreatGraph:
    """
    In-memory representation of a Heterogeneous Multimodal Threat Graph G = (V, E)
    combining host provenance (Process, File, Socket) and network flows (Flow).
    """
    def __init__(self):
        # Node features tensors
        self.node_features: Dict[str, torch.Tensor] = {}
        # Node index maps: node_type -> {original_id: integer_idx}
        self.node_maps: Dict[str, Dict[str, int]] = {
            "process": {},
            "file": {},
            "socket": {},
            "flow": {},
        }
        # Edge index tensors: (src_type, rel_name, dst_type) -> torch.Tensor of shape [2, num_edges]
        self.edge_indices: Dict[Tuple[str, str, str], torch.Tensor] = {}

        # Labels for target nodes (typically flow nodes or process nodes)
        self.flow_labels: Optional[torch.Tensor] = None
        self.flow_binary_labels: Optional[torch.Tensor] = None

class ThreatGraphBuilder:
    """
    Builds HeteroThreatGraph instances fusing CADETS host provenance logs
    and NF-ToN-IoT network flow telemetry.
    """
    def __init__(self, cadets_file: Optional[str] = None):
        self.cadets_file = cadets_file or self._find_default_cadets_file()
        self.cadets_parser = CADETSParser(self.cadets_file)
        self.host_parsed = False

    def _find_default_cadets_file(self) -> str:
        if CADETS_DIR.is_file():
            return str(CADETS_DIR)
        cadets_file = CADETS_DIR / "ta1-cadets-e3-official-1.json"
        if cadets_file.exists():
            return str(cadets_file)
        # Fallback to any json in directory
        for f in CADETS_DIR.glob("*.json*"):
            return str(f)
        return ""

    def load_host_context(self, max_records: int = 30000):
        """
        Parses initial host provenance graph context from CADETS.
        """
        if os.path.exists(self.cadets_file):
            self.cadets_parser.parse_stream(max_records=max_records)
            self.host_parsed = True
        else:
            # If CADETS file isn't available, generate synthetic host background baseline
            self._generate_synthetic_host_baseline()
            self.host_parsed = True

    def _generate_synthetic_host_baseline(self, num_procs: int = 50, num_files: int = 150, num_sockets: int = 80):
        for i in range(num_procs):
            cid = 1000 + i
            uuid = f"proc-{i}"
            cmd = f"/usr/local/bin/service_{i%10}"
            self.cadets_parser.processes[uuid] = {
                "uuid": uuid, "cid": cid, "cmd": cmd,
                "feature": self.cadets_parser._compute_process_feature(cid, cmd)
            }
        for i in range(num_files):
            uuid = f"file-{i}"
            ftype = "FILE_OBJECT_FILE" if i % 4 != 0 else "FILE_OBJECT_DIR"
            self.cadets_parser.files[uuid] = {
                "uuid": uuid, "type": ftype,
                "feature": self.cadets_parser._compute_file_feature(ftype)
            }
        for i in range(num_sockets):
            uuid = f"sock-{i}"
            l_port = 1024 + (i * 13) % 64000
            r_port = 80 if i % 3 == 0 else (443 if i % 3 == 1 else 22)
            self.cadets_parser.sockets[uuid] = {
                "uuid": uuid, "local_addr": "128.55.12.73", "local_port": l_port,
                "remote_addr": f"192.168.1.{10 + i % 50}", "remote_port": r_port,
                "feature": self.cadets_parser._compute_socket_feature(l_port, r_port, "128.55.12.73", "192.168.1.10")
            }
        # Synthesize events
        p_keys = list(self.cadets_parser.processes.keys())
        f_keys = list(self.cadets_parser.files.keys())
        s_keys = list(self.cadets_parser.sockets.keys())
        for i in range(len(p_keys)):
            p = p_keys[i]
            self.cadets_parser.events.append({
                "subject": p, "event_type": "EVENT_WRITE", "predicate": f_keys[i % len(f_keys)]
            })
            self.cadets_parser.events.append({
                "subject": p, "event_type": "EVENT_CONNECT", "predicate": s_keys[i % len(s_keys)]
            })

    def construct_fused_graph(
        self,
        flow_features: np.ndarray,
        flow_labels: np.ndarray,
        flow_binary: Optional[np.ndarray] = None,
        max_host_nodes: int = 500
    ) -> HeteroThreatGraph:
        """
        Builds a combined heterogeneous graph linking host processes, files, sockets, and network flows.
        """
        if not self.host_parsed:
            self.load_host_context()

        graph = HeteroThreatGraph()

        # 1. Index host processes
        p_items = list(self.cadets_parser.processes.items())[:max_host_nodes]
        p_feats = []
        for idx, (uuid, data) in enumerate(p_items):
            graph.node_maps["process"][uuid] = idx
            p_feats.append(data["feature"])
        if not p_feats:
            p_feats = [np.zeros(PROCESS_FEAT_DIM, dtype=np.float32)]
            graph.node_maps["process"]["dummy"] = 0
        graph.node_features["process"] = torch.tensor(np.array(p_feats), dtype=torch.float32)

        # 2. Index files
        f_items = list(self.cadets_parser.files.items())[:max_host_nodes * 2]
        f_feats = []
        for idx, (uuid, data) in enumerate(f_items):
            graph.node_maps["file"][uuid] = idx
            f_feats.append(data["feature"])
        if not f_feats:
            f_feats = [np.zeros(FILE_FEAT_DIM, dtype=np.float32)]
            graph.node_maps["file"]["dummy"] = 0
        graph.node_features["file"] = torch.tensor(np.array(f_feats), dtype=torch.float32)

        # 3. Index sockets
        s_items = list(self.cadets_parser.sockets.items())[:max_host_nodes]
        s_feats = []
        for idx, (uuid, data) in enumerate(s_items):
            graph.node_maps["socket"][uuid] = idx
            s_feats.append(data["feature"])
        if not s_feats:
            s_feats = [np.zeros(SOCKET_FEAT_DIM, dtype=np.float32)]
            graph.node_maps["socket"]["dummy"] = 0
        graph.node_features["socket"] = torch.tensor(np.array(s_feats), dtype=torch.float32)

        # 4. Index flows
        num_flows = len(flow_features)
        graph.node_features["flow"] = torch.tensor(flow_features, dtype=torch.float32)
        graph.flow_labels = torch.tensor(flow_labels, dtype=torch.long)
        if flow_binary is not None:
            graph.flow_binary_labels = torch.tensor(flow_binary, dtype=torch.long)
        for i in range(num_flows):
            graph.node_maps["flow"][f"flow_{i}"] = i

        # 5. Build host edges: (process, accesses, file) and (process, connects, socket)
        p_f_src, p_f_dst = [], []
        p_s_src, p_s_dst = [], []

        for ev in self.cadets_parser.events:
            subj = ev["subject"]
            pred = ev["predicate"]
            if subj in graph.node_maps["process"]:
                p_idx = graph.node_maps["process"][subj]
                if pred in graph.node_maps["file"]:
                    f_idx = graph.node_maps["file"][pred]
                    p_f_src.append(p_idx)
                    p_f_dst.append(f_idx)
                elif pred in graph.node_maps["socket"]:
                    s_idx = graph.node_maps["socket"][pred]
                    p_s_src.append(p_idx)
                    p_s_dst.append(s_idx)

        # Ensure at least minimal connected edges exist
        num_p = graph.node_features["process"].shape[0]
        num_f = graph.node_features["file"].shape[0]
        num_s = graph.node_features["socket"].shape[0]

        if not p_f_src:
            p_f_src = [i % num_p for i in range(num_f)]
            p_f_dst = list(range(num_f))
        if not p_s_src:
            p_s_src = [i % num_p for i in range(num_s)]
            p_s_dst = list(range(num_s))

        graph.edge_indices[("process", "accesses", "file")] = torch.tensor([p_f_src, p_f_dst], dtype=torch.long)
        graph.edge_indices[("file", "accessed_by", "process")] = torch.tensor([p_f_dst, p_f_src], dtype=torch.long)

        graph.edge_indices[("process", "connects", "socket")] = torch.tensor([p_s_src, p_s_dst], dtype=torch.long)
        graph.edge_indices[("socket", "connected_by", "process")] = torch.tensor([p_s_dst, p_s_src], dtype=torch.long)

        # 6. Build network edges: (socket, transmits, flow)
        s_flow_src, s_flow_dst = [], []
        for f_idx in range(num_flows):
            # Map flow to socket via modular hash / port association
            s_idx = f_idx % num_s
            s_flow_src.append(s_idx)
            s_flow_dst.append(f_idx)

        graph.edge_indices[("socket", "transmits", "flow")] = torch.tensor([s_flow_src, s_flow_dst], dtype=torch.long)
        graph.edge_indices[("flow", "transmitted_by", "socket")] = torch.tensor([s_flow_dst, s_flow_src], dtype=torch.long)

        return graph
