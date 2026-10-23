import os
import sys
import time
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Ensure root directory is on python path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    SAVED_MODELS_DIR,
    ATTACK_CLASSES,
    CLASS_TO_IDX,
    NETFLOW_FEATURE_COLS,
)
from src.realtime.engine import RealtimeDetectionEngine
from src.realtime.live_sniffer import LiveSystemFlowSniffer
from src.data.netflow_loader import NetFlowLoader
from src.federated.coordinator import FederatedCoordinator

app = FastAPI(title="SFM-GNN+ Real-Time SOC Interface")

# Initialize Engine, Live Sniffer, and Federated Coordinator
engine = RealtimeDetectionEngine()
live_sniffer = LiveSystemFlowSniffer()
fed_coordinator = FederatedCoordinator(num_clients=3)
netflow_loader = NetFlowLoader()

# Lazy / Cached sample pool for simulation mode
sample_pool = None
def get_sample_pool():
    global sample_pool
    if sample_pool is None:
        try:
            cache_file = SAVED_MODELS_DIR / "sample_pool_cache.parquet"
            if cache_file.exists():
                import pandas as pd
                sample_pool = pd.read_parquet(cache_file)
            else:
                sample_pool = netflow_loader.load_stratified_sample(samples_per_class=30, rare_class_boost=True)
                sample_pool.to_parquet(cache_file)
        except Exception as e:
            print(f"[-] Sample pool note: {e}")
    return sample_pool

# Global State
quarantined_records: List[Dict[str, Any]] = []
stream_active = True
stream_mode = "live_system" # "live_system" (Real Windows traffic) or "dataset_simulation"
stream_delay_seconds = 1.0
stats = {
    "total_processed": 0,
    "attacks_confirmed": 0,
    "fp_suppressed": 0,
    "normal_allowed": 0,
    "active_quarantines": 0,
}

class InjectionRequest(BaseModel):
    attack_type: str # e.g. "ransomware", "backdoor", "scanning", "Benign", etc.
    src_ip: Optional[str] = "192.168.1.185"
    dst_port: Optional[int] = 445

class ReleaseRequest(BaseModel):
    quarantine_id: int

class ModeRequest(BaseModel):
    mode: str # "live_system" or "dataset_simulation"

class FedRoundRequest(BaseModel):
    poison_client_id: Optional[int] = None # e.g. 2 to test ZK rejection

@app.get("/api/status")
async def get_status():
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return {
        "engine_ready": engine.is_ready,
        "device": cuda_name,
        "cuda_active": torch.cuda.is_available(),
        "model_architecture": "SFM-GNN+ (Heterogeneous GAT + Penultimate Verifier)",
        "model_accuracy": "92.58%",
        "stream_active": stream_active,
        "stream_mode": stream_mode,
        "stream_delay_seconds": stream_delay_seconds,
        "stats": {
            **stats,
            "active_quarantines": len(quarantined_records)
        }
    }

@app.get("/api/quarantine")
async def get_quarantine():
    return quarantined_records

@app.post("/api/quarantine/release")
async def release_quarantine(req: ReleaseRequest):
    global quarantined_records
    found = False
    for item in quarantined_records:
        if item.get("id") == req.quarantine_id:
            item["status"] = "RELEASED"
            found = True
            break
    quarantined_records = [q for q in quarantined_records if q.get("status") != "RELEASED"]
    stats["active_quarantines"] = len(quarantined_records)
    return {"success": found, "remaining_quarantined": len(quarantined_records)}

@app.post("/api/stream/mode")
async def set_stream_mode(req: ModeRequest):
    global stream_mode
    if req.mode in ["live_system", "dataset_simulation"]:
        stream_mode = req.mode
    return {"stream_mode": stream_mode}

@app.post("/api/stream/toggle")
async def toggle_stream():
    global stream_active
    stream_active = not stream_active
    return {"stream_active": stream_active}

@app.get("/api/federated/status")
async def get_federated_status():
    return {
        "current_round": fed_coordinator.current_round,
        "num_clients": fed_coordinator.num_clients,
        "clients": [
            {"id": c.node_id, "name": c.name, "public_key": str(c.get_public_key())[:12] + "..."}
            for c in fed_coordinator.clients
        ],
        "latest_round": fed_coordinator.round_history[-1] if fed_coordinator.round_history else None,
        "protocol": "Blind SecAgg (Zero-Sum Noise) + ZK-SNARK Gradient Bounds"
    }

@app.post("/api/federated/round")
async def trigger_federated_round(req: FedRoundRequest):
    """
    Executes a real Federated Learning round with SecAgg and ZK verification.
    """
    result = fed_coordinator.run_federated_round(poisoned_client_id=req.poison_client_id)
    return result

@app.post("/api/inject")
async def inject_flow(req: InjectionRequest):
    """
    Manually injects an attack flow into the GNN pipeline to test real-time detection & SDN response.
    """
    target_class = req.attack_type
    matched_row = None
    pool = get_sample_pool()
    if pool is not None and not pool.empty:
        match = pool[pool["Attack"] == target_class]
        if not match.empty:
            matched_row = match.sample(n=1).iloc[0]

    if matched_row is not None:
        raw_feat = matched_row[NETFLOW_FEATURE_COLS].values.reshape(1, -1)
    else:
        raw_feat = np.zeros((1, 41), dtype=np.float32)
        if target_class != "Benign":
            raw_feat[0, 8] = 0x18
            raw_feat[0, 4] = 15000

    meta = [{
        "src_ip": req.src_ip or "192.168.1.199",
        "cid": 1000 + int(np.random.randint(100, 9999)),
        "dst_port": req.dst_port or 445,
        "process_name": f"injected_{target_class.lower()}",
        "flow_id": f"INJECT-{int(time.time()*1000)%10000}"
    }]

    results = engine.process_flow_batch(raw_feat, meta)
    res = results[0]
    _update_state_from_result(res, meta[0])
    return res

def _update_state_from_result(res: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    stats["total_processed"] += 1
    verif = res.get("verification", {})
    status = verif.get("quality_status", "")

    if verif.get("is_confirmed_attack"):
        stats["attacks_confirmed"] += 1
        q_id = int(time.time() * 1000)
        q_entry = {
            "id": q_id,
            "timestamp": time.strftime("%H:%M:%S"),
            "threat_type": res.get("predicted_class_name", "Threat"),
            "confidence": f"{res.get('confidence', 0.0)*100:.1f}%",
            "src_ip": meta.get("src_ip", "10.0.0.1"),
            "dst_port": meta.get("dst_port", 80),
            "pid": meta.get("cid", 1024),
            "process_name": meta.get("process_name", "Unknown"),
            "openflow_rule": res.get("mitigation", {}).get("ovs_cli_command", ""),
            "status": "QUARANTINED"
        }
        quarantined_records.insert(0, q_entry)
        if len(quarantined_records) > 30:
            quarantined_records.pop()
    elif status == "FALSE_POSITIVE_SUPPRESSED":
        stats["fp_suppressed"] += 1
    else:
        stats["normal_allowed"] += 1

    stats["active_quarantines"] = len(quarantined_records)
    return res

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    print("[+] WebSocket client connected to real-time SOC stream.")
    
    flow_counter = 1
    try:
        while True:
            if not stream_active:
                await asyncio.sleep(0.5)
                continue

            # Determine traffic source: Live Windows System vs Dataset Simulation
            if stream_mode == "live_system":
                # Real active Windows traffic & processes!
                live_flows = live_sniffer.capture_live_system_flows(max_flows=1)
                if live_flows:
                    flow_obj = live_flows[0]
                    raw_flow = flow_obj["features"].reshape(1, -1)
                    src_ip = flow_obj["src_ip"]
                    dst_port = flow_obj["dst_port"]
                    cid = flow_obj["pid"]
                    pname = flow_obj["process_name"]
                    ground_truth = "Live System Traffic"
                else:
                    # Fallback to local loopback socket
                    raw_flow = np.zeros((1, 41), dtype=np.float32)
                    raw_flow[0, 8] = 0x10
                    src_ip = "127.0.0.1"
                    dst_port = 8000
                    cid = os.getpid()
                    pname = "python.exe"
                    ground_truth = "Live System Traffic"
            else:
                # Benchmark dataset simulation mode
                pool = get_sample_pool()
                if pool is not None and not pool.empty:
                    sample_row = pool.sample(n=1).iloc[0]
                    raw_flow = sample_row[NETFLOW_FEATURE_COLS].values.reshape(1, -1)
                    ground_truth = str(sample_row["Attack"])
                else:
                    raw_flow = np.zeros((1, 41), dtype=np.float32)
                    ground_truth = "Benign"

                cid = 1000 + (flow_counter * 37) % 8000
                pname = f"service_{cid % 12}.exe"
                src_ip = f"192.168.1.{100 + (flow_counter % 30)}"
                dst_port = int(raw_flow[0, 1]) if raw_flow[0, 1] > 0 else 80

            meta = [{
                "src_ip": src_ip,
                "cid": cid,
                "dst_port": dst_port,
                "process_name": pname,
                "flow_id": f"FLOW-{flow_counter:04d}",
                "mode": stream_mode
            }]

            results = engine.process_flow_batch(raw_flow, meta)
            res = results[0]
            res["ground_truth"] = ground_truth
            res["metadata"] = meta[0]

            _update_state_from_result(res, meta[0])

            # Build localized Heterogeneous Threat Graph Window
            threat_name = res.get("predicted_class_name", "Benign")
            is_attack = res.get("verification", {}).get("is_confirmed_attack", False)
            
            graph_payload = {
                "nodes": [
                    {"id": f"proc_{cid}", "label": f"{pname} (PID {cid})", "type": "process", "color": "#38bdf8"},
                    {"id": f"sock_{src_ip}", "label": f"{src_ip}:{dst_port}", "type": "socket", "color": "#a855f7"},
                    {"id": f"file_{cid}", "label": f"/proc/{cid}/exe", "type": "file", "color": "#94a3b8"},
                    {"id": f"flow_{flow_counter}", "label": f"Flow {meta[0]['flow_id']} ({threat_name})", "type": "flow", "color": "#ef4444" if is_attack else "#10b981"}
                ],
                "edges": [
                    {"source": f"proc_{cid}", "target": f"file_{cid}", "relation": "executes"},
                    {"source": f"proc_{cid}", "target": f"sock_{src_ip}", "relation": "binds"},
                    {"source": f"sock_{src_ip}", "target": f"flow_{flow_counter}", "relation": "transmits"}
                ]
            }
            res["subgraph"] = graph_payload

            # Transmit to frontend WebSocket
            await websocket.send_json({
                "type": "FLOW_EVENT",
                "data": res,
                "stats": {
                    **stats,
                    "active_quarantines": len(quarantined_records)
                }
            })

            flow_counter += 1
            await asyncio.sleep(stream_delay_seconds)

    except WebSocketDisconnect:
        print("[-] WebSocket client disconnected.")
    except Exception as e:
        print(f"[-] WebSocket stream error: {e}")

# Mount static frontend files
FRONTEND_DIR = BASE_DIR / "web"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    print("[*] Starting SFM-GNN+ SOC Server on http://127.0.0.1:8000")
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=False)
