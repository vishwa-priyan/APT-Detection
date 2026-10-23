import os
import json
import hashlib
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from src.config import PROCESS_FEAT_DIM, FILE_FEAT_DIM, SOCKET_FEAT_DIM

class CADETSParser:
    """
    High-performance streaming parser for DARPA TC E3 CADETS CDM-18 JSON audit logs.
    Extracts Process, File, Socket entities and causal execution events.
    """
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath
        self.processes: Dict[str, Dict[str, Any]] = {}
        self.files: Dict[str, Dict[str, Any]] = {}
        self.sockets: Dict[str, Dict[str, Any]] = {}
        self.events: List[Dict[str, Any]] = []

    def parse_stream(self, max_records: int = 50000) -> Dict[str, Any]:
        """
        Parses a stream of CDM JSON records from the CADETS file.
        """
        if not self.filepath or not os.path.exists(self.filepath):
            raise FileNotFoundError(f"CADETS log file not found at: {self.filepath}")

        records_parsed = 0
        with open(self.filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if records_parsed >= max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    datum = record.get("datum", {})
                    for key, val in datum.items():
                        entity_type = key.split(".")[-1]
                        self._process_datum(entity_type, val)
                    records_parsed += 1
                except Exception:
                    continue

        return {
            "num_processes": len(self.processes),
            "num_files": len(self.files),
            "num_sockets": len(self.sockets),
            "num_events": len(self.events),
        }

    def _process_datum(self, entity_type: str, val: Dict[str, Any]):
        uuid = val.get("uuid")
        if not uuid:
            return

        if entity_type == "Subject":
            cid = val.get("cid", 0)
            local_principal = val.get("localPrincipal")
            cmd_dict = val.get("cmdLine")
            cmd = ""
            if isinstance(cmd_dict, dict):
                cmd = cmd_dict.get("string", "")
            
            self.processes[uuid] = {
                "uuid": uuid,
                "cid": cid,
                "cmd": cmd,
                "principal": local_principal,
                "feature": self._compute_process_feature(cid, cmd)
            }

        elif entity_type == "FileObject":
            ftype = val.get("type", "FILE_OBJECT_FILE")
            self.files[uuid] = {
                "uuid": uuid,
                "type": ftype,
                "feature": self._compute_file_feature(ftype)
            }

        elif entity_type == "NetFlowObject":
            l_addr = str(val.get("localAddress", ""))
            l_port = int(val.get("localPort") or 0)
            r_addr = str(val.get("remoteAddress", ""))
            r_port = int(val.get("remotePort") or 0)
            self.sockets[uuid] = {
                "uuid": uuid,
                "local_addr": l_addr,
                "local_port": l_port,
                "remote_addr": r_addr,
                "remote_port": r_port,
                "feature": self._compute_socket_feature(l_port, r_port, l_addr, r_addr)
            }

        elif entity_type == "Event":
            ev_type = val.get("type", "")
            subj = (val.get("subject") or {}).get("com.bbn.tc.schema.avro.cdm18.UUID")
            pred = (val.get("predicateObject") or {}).get("com.bbn.tc.schema.avro.cdm18.UUID")
            if subj and pred:
                self.events.append({
                    "subject": subj,
                    "event_type": ev_type,
                    "predicate": pred,
                    "timestamp": val.get("timestampNanos", 0)
                })

    def _compute_process_feature(self, cid: int, cmd: str) -> np.ndarray:
        feat = np.zeros(PROCESS_FEAT_DIM, dtype=np.float32)
        # Process CID norm
        feat[0] = float(cid % 65535) / 65535.0
        # Command length & entropy
        feat[1] = min(len(cmd) / 256.0, 1.0)
        # Hash features for executable
        if cmd:
            h = int(hashlib.md5(cmd.encode("utf-8")).hexdigest()[:8], 16)
            for idx in range(2, PROCESS_FEAT_DIM):
                feat[idx] = ((h >> (idx * 2)) & 0x0F) / 15.0
        return feat

    def _compute_file_feature(self, ftype: str) -> np.ndarray:
        feat = np.zeros(FILE_FEAT_DIM, dtype=np.float32)
        types = ["FILE_OBJECT_FILE", "FILE_OBJECT_DIR", "FILE_OBJECT_CHAR", "FILE_OBJECT_BLOCK", "FILE_OBJECT_SOCKET"]
        for idx, t in enumerate(types):
            if t == ftype and idx < FILE_FEAT_DIM:
                feat[idx] = 1.0
        return feat

    def _compute_socket_feature(self, l_port: int, r_port: int, l_addr: str, r_addr: str) -> np.ndarray:
        feat = np.zeros(SOCKET_FEAT_DIM, dtype=np.float32)
        feat[0] = min(l_port / 65535.0, 1.0)
        feat[1] = min(r_port / 65535.0, 1.0)
        # Well-known / privileged ports indicator
        feat[2] = 1.0 if l_port < 1024 else 0.0
        feat[3] = 1.0 if r_port < 1024 else 0.0
        # IP locality indicator (private vs public IP)
        feat[4] = 1.0 if l_addr.startswith(("10.", "192.168.", "172.", "127.", "128.")) else 0.0
        feat[5] = 1.0 if r_addr.startswith(("10.", "192.168.", "172.", "127.", "128.")) else 0.0
        # Port entropy
        feat[6] = ((l_port ^ r_port) & 0xFF) / 255.0
        feat[7] = 1.0
        return feat
