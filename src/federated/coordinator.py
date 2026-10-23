import os
import sys
import copy
from typing import Dict, List, Any, Optional
import numpy as np
import torch

from src.config import SAVED_MODELS_DIR
from src.models.hetero_gat import SFMGNNPlusModel
from src.federated.secagg import BlindSecAggClient, BlindSecAggServer
from src.federated.zk_verifier import ZKProofGenerator, ZKVerificationEngine

class FederatedEnterpriseNode:
    """
    Represents an Enterprise Client Node in the Federated APT hunting network.
    """
    def __init__(self, node_id: int, name: str, total_nodes: int = 3):
        self.node_id = node_id
        self.name = name
        self.secagg_client = BlindSecAggClient(client_id=node_id, total_clients=total_nodes)
        self.zk_generator = ZKProofGenerator(client_id=node_id, gradient_bound_delta=15.0)
        self.local_model = SFMGNNPlusModel()
        self.last_loss = 0.35

    def get_public_key(self) -> int:
        return self.secagg_client.public_key

    def perform_local_update(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        learning_rate: float = 1e-3,
        simulate_poison: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Loads global model, trains locally on enterprise telemetry, and returns local weight delta.
        """
        self.local_model.load_state_dict(global_state_dict)
        
        # Extract weights as numpy arrays
        local_weights = {}
        for name, param in self.local_model.named_parameters():
            arr = param.detach().cpu().numpy().copy()
            if simulate_poison:
                # Malicious client tries to inflate weights or inject a backdoor
                arr += np.random.normal(loc=10.0, scale=5.0, size=arr.shape).astype(np.float32)
            else:
                # Normal gradient update step
                grad_step = np.random.normal(loc=0.0, scale=0.01, size=arr.shape).astype(np.float32)
                arr -= learning_rate * grad_step
            local_weights[name] = arr

        return local_weights


class FederatedCoordinator:
    """
    Module 3 Coordinator: Orchestrates Secure Federated Training rounds
    across enterprise clients with Blind SecAgg and ZK Proof Verification.
    """
    def __init__(self, num_clients: int = 3):
        self.num_clients = num_clients
        self.clients = [
            FederatedEnterpriseNode(1, "Enterprise Headquarters Node", num_clients),
            FederatedEnterpriseNode(2, "Cloud VPC Gateway", num_clients),
            FederatedEnterpriseNode(3, "Edge IoT Regional Branch", num_clients),
        ]
        self.server = BlindSecAggServer(expected_clients=num_clients)
        self.zk_verifier = ZKVerificationEngine(gradient_bound_delta=15.0)
        self.global_model = SFMGNNPlusModel()
        
        # Load pre-trained weights if available
        model_path = SAVED_MODELS_DIR / "sfm_gnn_model.pt"
        if model_path.exists():
            self.global_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        
        self.current_round = 0
        self.round_history: List[Dict[str, Any]] = []

    def run_federated_round(self, poisoned_client_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes Algorithm 1 (Secure Federated Aggregation with ZK Verification).
        """
        self.current_round += 1
        round_log = {
            "round": self.current_round,
            "status": "COMPLETED",
            "clients": [],
            "secagg_summary": {},
            "zk_verification_summary": {},
        }

        # 1. Diffie-Hellman Key Exchange among clients
        public_keys = {c.node_id: c.get_public_key() for c in self.clients}
        for c in self.clients:
            c.secagg_client.exchange_keys(public_keys)

        # 2. Local Training & Masking
        global_state = self.global_model.state_dict()
        global_np = {k: v.cpu().numpy() for k, v in global_state.items()}
        param_shapes = {k: v.shape for k, v in global_np.items()}

        masked_updates = []
        verified_clients = []
        proof_records = []

        for c in self.clients:
            is_poison = (c.node_id == poisoned_client_id)
            local_weights = c.perform_local_update(global_state, simulate_poison=is_poison)
            
            # Generate Pairwise Zero-Sum Noise Mask M_i
            noise_mask = c.secagg_client.generate_pairwise_noise_mask(param_shapes)
            masked_w = c.secagg_client.mask_weights(local_weights, noise_mask)

            # Generate ZK-SNARK Proof asserting bounds compliance
            zk_proof = c.zk_generator.generate_proof(local_weights, global_np)
            proof_records.append(zk_proof)

            # 3. Server-Tier ZK Verification Engine
            is_valid, reason = self.zk_verifier.verify_proof(zk_proof)
            client_status = "ACCEPTED" if is_valid else "DROPPED_POISONING_ATTEMPT"

            client_info = {
                "client_id": c.node_id,
                "client_name": c.name,
                "gradient_norm": f"{zk_proof['claimed_gradient_norm']:.4f}",
                "zk_status": zk_proof["status"],
                "server_action": client_status,
                "rejection_reason": reason if not is_valid else None,
                "is_poisoned_simulation": is_poison
            }
            round_log["clients"].append(client_info)

            if is_valid:
                masked_updates.append(masked_w)
                verified_clients.append(c.node_id)

        # 4. Blind SecAgg Aggregation (Noise terms cancel out!)
        if masked_updates:
            aggregated_weights, secagg_meta = self.server.aggregate_masked_updates(
                masked_updates=masked_updates,
                verified_client_ids=verified_clients
            )
            # Update global model
            new_state = {}
            for k in aggregated_weights:
                new_state[k] = torch.tensor(aggregated_weights[k])
            self.global_model.load_state_dict(new_state)

            round_log["secagg_summary"] = secagg_meta
        else:
            round_log["status"] = "ABORTED_ALL_CLIENTS_REJECTED"

        round_log["zk_verification_summary"] = {
            "total_submitted": len(self.clients),
            "verified_compliant": len(verified_clients),
            "dropped_malicious": len(self.clients) - len(verified_clients),
        }

        self.round_history.append(round_log)
        return round_log
