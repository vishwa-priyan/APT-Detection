import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure root directory is on python path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import SAVED_MODELS_DIR
from src.models.hetero_gat import SFMGNNPlusModel
from src.federated.secagg import BlindSecAggServer
from src.federated.zk_verifier import ZKVerificationEngine

app = FastAPI(title="SFM-GNN+ Independent Federated Aggregator Server")

class ClientRegistration(BaseModel):
    client_id: int
    name: str
    public_key: int
    endpoint: Optional[str] = None

class ClientSubmission(BaseModel):
    client_id: int
    round_number: int
    masked_weights: Dict[str, List[float]]  # flattened or shape-encoded weights
    param_shapes: Dict[str, List[int]]
    zk_proof: Dict[str, Any]

# Global Server State
current_round = 1
registered_clients: Dict[int, Dict[str, Any]] = {}
current_round_submissions: Dict[int, Dict[str, Any]] = {}
round_history: List[Dict[str, Any]] = []

server_aggregator = BlindSecAggServer(expected_clients=3)
zk_verifier = ZKVerificationEngine(gradient_bound_delta=15.0)
global_model = SFMGNNPlusModel()

# Load saved weights if available
model_path = SAVED_MODELS_DIR / "sfm_gnn_model.pt"
if model_path.exists():
    try:
        global_model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    except Exception:
        try:
            global_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        except Exception as e:
            print(f"[-] Model load note: {e}")

@app.get("/")
async def root():
    return {
        "service": "SFM-GNN+ Independent Federated Server",
        "current_round": current_round,
        "registered_clients_count": len(registered_clients),
        "submissions_received": len(current_round_submissions)
    }

@app.post("/federated/register")
async def register_client(reg: ClientRegistration):
    """
    Independent client registers with its Diffie-Hellman public key component.
    """
    registered_clients[reg.client_id] = {
        "client_id": reg.client_id,
        "name": reg.name,
        "public_key": reg.public_key,
        "registered_at": time.time(),
        "endpoint": reg.endpoint
    }
    print(f"[+] Client Registered: ID={reg.client_id} ({reg.name}) with PK={str(reg.public_key)[:10]}...")
    return {
        "status": "REGISTERED",
        "client_id": reg.client_id,
        "current_round": current_round,
        "total_registered": len(registered_clients)
    }

@app.get("/federated/peers")
async def get_peer_public_keys():
    """
    Returns public keys of all registered clients so peers can compute
    pairwise Diffie-Hellman shared secrets for zero-sum noise masking.
    """
    return {
        cid: info["public_key"] for cid, info in registered_clients.items()
    }

@app.get("/federated/global_weights")
async def get_global_weights():
    """
    Returns the current global model weights for local training.
    """
    state_dict = global_model.state_dict()
    weights_json = {}
    shapes_json = {}
    for k, v in state_dict.items():
        arr = v.cpu().numpy()
        weights_json[k] = arr.flatten().tolist()
        shapes_json[k] = list(arr.shape)
    return {
        "round_number": current_round,
        "weights": weights_json,
        "shapes": shapes_json
    }

@app.post("/federated/submit_update")
async def submit_update(sub: ClientSubmission):
    """
    Receives masked weights (W_i + M_i) and ZK proof from an independent client node.
    """
    if sub.round_number != current_round:
        raise HTTPException(status_code=400, detail=f"Submission is for round {sub.round_number}, but server is on round {current_round}")

    # Reconstruct numpy arrays
    reconstructed_masked = {}
    for name, flat_list in sub.masked_weights.items():
        shape = sub.param_shapes[name]
        reconstructed_masked[name] = np.array(flat_list, dtype=np.float32).reshape(shape)

    current_round_submissions[sub.client_id] = {
        "client_id": sub.client_id,
        "masked_weights": reconstructed_masked,
        "zk_proof": sub.zk_proof,
        "received_at": time.time()
    }

    print(f"[+] Received masked update from Client {sub.client_id} for Round {current_round}")
    return {
        "status": "RECEIVED",
        "submissions_count": len(current_round_submissions),
        "expected_clients": len(registered_clients)
    }

@app.post("/federated/aggregate_round")
async def trigger_aggregation():
    """
    Aggregates all submissions for the current round using ZK verification and Blind SecAgg.
    """
    global current_round, current_round_submissions, round_history

    if not current_round_submissions:
        raise HTTPException(status_code=400, detail="No submissions to aggregate.")

    verified_masked_updates = []
    verified_client_ids = []
    dropped_records = []

    print(f"\n[*] Executing Aggregation for Round {current_round}...")

    # 1. ZK-SNARK Bounds Verification for each client
    for cid, submission in current_round_submissions.items():
        proof = submission["zk_proof"]
        is_valid, reason = zk_verifier.verify_proof(proof)

        if is_valid:
            print(f"  [OK] Client {cid}: ZK Proof VALID (Gradient Norm: {proof.get('claimed_gradient_norm', 0.0):.4f} <= bound). Update Accepted.")
            verified_masked_updates.append(submission["masked_weights"])
            verified_client_ids.append(cid)
        else:
            print(f"  [REJECT] Client {cid}: ZK Proof REJECTED ({reason})! Update dropped from aggregation.")
            dropped_records.append({
                "client_id": cid,
                "reason": reason,
                "claimed_norm": proof.get("claimed_gradient_norm")
            })

    # 2. Blind SecAgg Aggregation (Pairwise noise terms cancel out!)
    if verified_masked_updates:
        aggregated_weights, secagg_meta = server_aggregator.aggregate_masked_updates(
            masked_updates=verified_masked_updates,
            verified_client_ids=verified_client_ids
        )
        # Update global model
        new_state = {}
        for k in aggregated_weights:
            new_state[k] = torch.tensor(aggregated_weights[k])
        global_model.load_state_dict(new_state)

        round_summary = {
            "round": current_round,
            "status": "COMPLETED",
            "participating_clients": verified_client_ids,
            "dropped_clients": dropped_records,
            "secagg_summary": secagg_meta,
            "completed_at": time.time()
        }
    else:
        round_summary = {
            "round": current_round,
            "status": "FAILED_ALL_CLIENTS_REJECTED",
            "participating_clients": [],
            "dropped_clients": dropped_records,
            "completed_at": time.time()
        }

    round_history.append(round_summary)
    current_round += 1
    current_round_submissions = {}

    print(f"[+] Round Complete! Progressed to Round {current_round}.\n")
    return round_summary

@app.get("/federated/status")
async def get_federated_status():
    return {
        "current_round": current_round,
        "registered_clients": list(registered_clients.values()),
        "pending_submissions": list(current_round_submissions.keys()),
        "round_history": round_history[-5:] if round_history else []
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("FED_SERVER_PORT", 8001))
    print(f"[*] Starting Independent Federated Aggregator Server on http://0.0.0.0:{port}")
    uvicorn.run("src.federated.server_node:app", host="0.0.0.0", port=port, reload=False)
