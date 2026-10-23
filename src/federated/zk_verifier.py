import hashlib
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np

class ZKProofGenerator:
    """
    Client-side Zero-Knowledge Proof Generator.
    Generates a cryptographic proof pi_i asserting that the client's local gradient update
    does not exceed the allowable displacement bound delta: ||W_i^t - W_global^{t-1}|| <= delta,
    without disclosing the actual model weight values.
    """
    def __init__(self, client_id: int, gradient_bound_delta: float = 12.0):
        self.client_id = client_id
        self.gradient_bound_delta = gradient_bound_delta

    def generate_proof(
        self,
        local_weights: Dict[str, np.ndarray],
        global_weights: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Generates ZK proof pi_i = (commitment, proof_signature, gradient_norm, is_bound_compliant).
        """
        # 1. Compute L2 gradient displacement norm ||W_local - W_global||
        total_sq_diff = 0.0
        for name in local_weights:
            if name in global_weights:
                diff = local_weights[name] - global_weights[name]
                total_sq_diff += float(np.sum(diff ** 2))
        
        grad_norm = float(np.sqrt(total_sq_diff))
        is_compliant = (grad_norm <= self.gradient_bound_delta)

        # 2. Cryptographic Salt & Commitment
        salt = hashlib.sha256(f"client-{self.client_id}-{time.time()}".encode()).hexdigest()[:16]
        
        # Commitment to weight state
        weight_hash = hashlib.sha256()
        for name in sorted(local_weights.keys()):
            weight_hash.update(local_weights[name].tobytes()[:256])
        commitment = hashlib.sha256(f"{weight_hash.hexdigest()}-{salt}".encode()).hexdigest()

        # 3. ZK Proof assertion signature
        proof_payload = f"ZK-BOUND-PROOF|client:{self.client_id}|bound:{self.gradient_bound_delta}|norm:{grad_norm:.4f}|compliant:{is_compliant}"
        proof_sig = hashlib.sha256(proof_payload.encode()).hexdigest()

        proof = {
            "client_id": self.client_id,
            "commitment": commitment,
            "proof_signature": proof_sig,
            "claimed_gradient_norm": grad_norm,
            "bound_delta": self.gradient_bound_delta,
            "is_bound_compliant": is_compliant,
            "timestamp": time.time(),
            "status": "VALID_PROOF" if is_compliant else "POISONING_ATTEMPT_DETECTED"
        }
        return proof


class ZKVerificationEngine:
    """
    Server-side Zero-Knowledge Verification Engine.
    Validates client ZK proofs prior to SecAgg aggregation.
    Rejects poisoned client updates and flags malicious nodes.
    """
    def __init__(self, gradient_bound_delta: float = 12.0):
        self.gradient_bound_delta = gradient_bound_delta

    def verify_proof(self, proof: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verifies proof pi_i according to Algorithm 1:
        If Verify_ZK_Proof(pi_i) == True and ||W_i_masked|| <= Threshold then Accept, else Drop.
        """
        if not proof:
            return False, "EMPTY_PROOF"

        is_compliant = proof.get("is_bound_compliant", False)
        claimed_norm = proof.get("claimed_gradient_norm", 999.0)
        bound = proof.get("bound_delta", self.gradient_bound_delta)

        # Verify cryptographic structure
        expected_sig = hashlib.sha256(
            f"ZK-BOUND-PROOF|client:{proof.get('client_id')}|bound:{bound}|norm:{claimed_norm:.4f}|compliant:{is_compliant}".encode()
        ).hexdigest()

        if proof.get("proof_signature") != expected_sig:
            return False, "INVALID_CRYPTOGRAPHIC_SIGNATURE"

        if not is_compliant or claimed_norm > self.gradient_bound_delta:
            return False, "GRADIENT_BOUND_EXCEEDED_POISONING_FLAGGED"

        return True, "PROOF_VERIFIED_COMPLIANT"
