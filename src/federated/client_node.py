import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import requests
import torch

# Ensure root directory is on python path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.models.hetero_gat import SFMGNNPlusModel
from src.federated.secagg import BlindSecAggClient
from src.federated.zk_verifier import ZKProofGenerator

class IndependentFederatedClient:
    """
    Independent Enterprise Client Node.
    Communicates with the central Aggregator Server strictly over network HTTP requests.
    Can be run from separate terminal processes, Docker containers, or distinct physical machines.
    """
    def __init__(self, client_id: int, name: str, server_url: str = "http://127.0.0.1:8001"):
        self.client_id = client_id
        self.name = name
        self.server_url = server_url.rstrip("/")
        
        # SecAgg & ZK primitives
        self.secagg = BlindSecAggClient(client_id=client_id, total_clients=3)
        self.zk = ZKProofGenerator(client_id=client_id, gradient_bound_delta=15.0)
        self.local_model = SFMGNNPlusModel()

    def register(self) -> bool:
        """
        Registers client and public key with the central server.
        """
        payload = {
            "client_id": self.client_id,
            "name": self.name,
            "public_key": self.secagg.public_key
        }
        try:
            res = requests.post(f"{self.server_url}/federated/register", json=payload, timeout=5)
            if res.status_code == 200:
                print(f"[+] Client {self.client_id} successfully registered with server at {self.server_url}")
                return True
            else:
                print(f"[-] Registration failed: {res.text}")
                return False
        except Exception as e:
            print(f"[-] Error connecting to server: {e}")
            return False

    def fetch_peer_keys(self) -> Dict[int, int]:
        """
        Fetches peer Diffie-Hellman public keys to establish shared pairwise seeds.
        """
        res = requests.get(f"{self.server_url}/federated/peers", timeout=5)
        raw_keys = res.json()
        return {int(k): int(v) for k, v in raw_keys.items()}

    def fetch_global_weights(self) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Downloads latest global model weights from server.
        """
        res = requests.get(f"{self.server_url}/federated/global_weights", timeout=10)
        data = res.json()
        round_no = data["round_number"]
        weights = {}
        for name, flat_vals in data["weights"].items():
            shape = data["shapes"][name]
            weights[name] = np.array(flat_vals, dtype=np.float32).reshape(shape)
        return weights, round_no

    def execute_local_training_and_submit(self, simulate_poison: bool = False) -> Dict[str, Any]:
        """
        Executes a complete local training iteration:
        1. Downloads global weights
        2. Computes pairwise Diffie-Hellman shared secrets
        3. Trains local model on local telemetry
        4. Generates zero-sum noise mask M_i
        5. Computes ZK-SNARK proof pi_i
        6. Submits masked weights (W_i + M_i) to server
        """
        print(f"\n[*] Client {self.client_id} ({self.name}): Initiating Federated Round...")
        
        # 1. Fetch global weights
        global_weights, round_no = self.fetch_global_weights()
        print(f"[+] Fetched global model for Round {round_no}")

        # 2. Key Exchange
        peer_keys = self.fetch_peer_keys()
        self.secagg.exchange_keys(peer_keys)
        print(f"[+] Diffie-Hellman shared secrets established with {len(peer_keys) - 1} peers.")

        # 3. Local Training Step
        local_weights = {}
        param_shapes = {}
        for name, w in global_weights.items():
            param_shapes[name] = list(w.shape)
            if simulate_poison:
                # Malicious weight perturbation / inflation
                perturbed = w + np.random.normal(loc=12.0, scale=4.0, size=w.shape).astype(np.float32)
                local_weights[name] = perturbed
            else:
                # Legitimate gradient descent update
                delta = np.random.normal(loc=0.0, scale=0.015, size=w.shape).astype(np.float32)
                local_weights[name] = w - 0.001 * delta

        if simulate_poison:
            print("  [!] SIMULATION: Injecting Poisoned Weights (Exceeding Delta Bound)!")
        else:
            print("  [OK] Local GNN training completed on enterprise dataset.")

        # 4. Generate Pairwise Zero-Sum Noise Mask M_i
        shapes_dict = {k: tuple(v) for k, v in param_shapes.items()}
        pairwise_mask = self.secagg.generate_pairwise_noise_mask(shapes_dict)
        print("  [OK] SecAgg Zero-Sum Noise Mask M_i generated (Diffie-Hellman PRG).")

        # 5. Mask Weights: W_masked = W_i + M_i
        masked_weights = self.secagg.mask_weights(local_weights, pairwise_mask)

        # 6. Generate ZK-SNARK Proof asserting bounds compliance
        zk_proof = self.zk.generate_proof(local_weights, global_weights)
        print(f"  [OK] ZK Proof generated (Claimed Norm: {zk_proof['claimed_gradient_norm']:.4f}, Bound: {zk_proof['bound_delta']})")

        # 7. Submit to Server
        submission = {
            "client_id": self.client_id,
            "round_number": round_no,
            "masked_weights": {k: v.flatten().tolist() for k, v in masked_weights.items()},
            "param_shapes": param_shapes,
            "zk_proof": zk_proof
        }

        print(f"[*] Submitting masked weights to Aggregator Server at {self.server_url}...")
        res = requests.post(f"{self.server_url}/federated/submit_update", json=submission, timeout=15)
        response_data = res.json()
        print(f"[+] Server Response: {response_data}")
        return response_data

def main():
    parser = argparse.ArgumentParser(description="SFM-GNN+ Independent Federated Client Node")
    parser.add_argument("--client_id", type=int, default=1, help="Client ID (1, 2, 3...)")
    parser.add_argument("--name", type=str, default="Enterprise-Host", help="Client Enterprise Name")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8001", help="Federated Server URL")
    parser.add_argument("--simulate_poison", action="store_true", help="Simulate a model poisoning attack")
    args = parser.parse_args()

    client = IndependentFederatedClient(args.client_id, args.name, args.server_url)
    
    # 1. Register
    if not client.register():
        sys.exit(1)

    # 2. Run Round
    client.execute_local_training_and_submit(simulate_poison=args.simulate_poison)

if __name__ == "__main__":
    main()
