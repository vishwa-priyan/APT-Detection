from typing import Dict, List, Any, Optional
import numpy as np
import torch
from src.config import NETFLOW_FEATURE_COLS, ATTACK_CLASSES

class ThreatGraphExplainer:
    """
    Local XAI Attribution Engine:
    Computes gradient sensitivity and feature importance rankings on the
    HeteroGAT model's flow and structural inputs, producing actionable incident explanations.
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.feature_names = NETFLOW_FEATURE_COLS

    def explain_flow(
        self,
        node_feats: Dict[str, torch.Tensor],
        edge_indices: Dict[Any, torch.Tensor],
        flow_idx: int,
        target_class: Optional[int] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Computes input gradient attribution for a specific flow node.
        """
        self.model.eval()
        
        # Clone and require grad for flow features
        flow_feats = node_feats["flow"].clone().detach().requires_grad_(True)
        feats_copy = {k: v.clone().detach() for k, v in node_feats.items()}
        feats_copy["flow"] = flow_feats

        logits, _ = self.model(feats_copy, edge_indices, return_penultimate=False)
        target_logits = logits[flow_idx]

        if target_class is None:
            target_class = int(torch.argmax(target_logits).item())

        score = target_logits[target_class]
        score.backward()

        grad = flow_feats.grad[flow_idx].detach().cpu().numpy()
        raw_val = node_feats["flow"][flow_idx].detach().cpu().numpy()

        # Attribution = input * gradient
        attribution = np.abs(grad * raw_val)
        
        # Top-k features
        top_indices = np.argsort(-attribution)[:top_k]
        top_features = []
        for idx in top_indices:
            top_features.append({
                "feature_name": self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}",
                "attribution_score": float(attribution[idx]),
                "raw_value": float(raw_val[idx]),
                "gradient_direction": "positive" if grad[idx] > 0 else "negative"
            })

        return {
            "target_class_idx": target_class,
            "target_class_name": ATTACK_CLASSES[target_class] if target_class < len(ATTACK_CLASSES) else "Unknown",
            "top_attributions": top_features,
            "flow_index": flow_idx
        }
