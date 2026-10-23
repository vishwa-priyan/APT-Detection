import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from src.config import (
    PROCESS_FEAT_DIM,
    FILE_FEAT_DIM,
    SOCKET_FEAT_DIM,
    FLOW_FEAT_DIM,
    HIDDEN_DIM,
    PENULTIMATE_DIM,
    NUM_CLASSES,
    NUM_HEADS,
    DROPOUT,
)

class RelationGATConv(nn.Module):
    """
    Multi-head Graph Attention Convolution for a specific directed edge relation (src -> dst).
    """
    def __init__(self, in_src: int, in_dst: int, out_dim: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.head_dim = out_dim // heads
        assert self.head_dim * heads == out_dim, "out_dim must be divisible by heads"

        self.lin_src = nn.Linear(in_src, out_dim, bias=False)
        self.lin_dst = nn.Linear(in_dst, out_dim, bias=False)
        
        self.attn_src = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(1, heads, self.head_dim))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x_src: [N_src, in_src]
        x_dst: [N_dst, in_dst]
        edge_index: [2, E] where edge_index[0] is src_idx and edge_index[1] is dst_idx
        """
        num_dst = x_dst.size(0)
        h_src = self.lin_src(x_src).view(-1, self.heads, self.head_dim)  # [N_src, heads, head_dim]
        h_dst = self.lin_dst(x_dst).view(-1, self.heads, self.head_dim)  # [N_dst, heads, head_dim]

        if edge_index.size(1) == 0:
            return torch.zeros(num_dst, self.out_dim, device=x_dst.device)

        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        # Gather node projections for each edge
        h_src_edge = h_src[src_idx]  # [E, heads, head_dim]
        h_dst_edge = h_dst[dst_idx]  # [E, heads, head_dim]

        # Compute raw attention coefficients: (a_src * h_src + a_dst * h_dst)
        score = (h_src_edge * self.attn_src).sum(dim=-1) + (h_dst_edge * self.attn_dst).sum(dim=-1)  # [E, heads]
        alpha = self.leaky_relu(score)

        # Softmax normalization over target dst_idx
        # Manual segmented softmax for stable computation
        alpha_exp = torch.exp(alpha - alpha.max(dim=0, keepdim=True)[0])
        denom = torch.zeros(num_dst, self.heads, device=x_dst.device)
        denom.index_add_(0, dst_idx, alpha_exp)
        denom = denom.clamp(min=1e-8)
        alpha_norm = alpha_exp / denom[dst_idx]
        alpha_norm = self.dropout(alpha_norm)  # [E, heads]

        # Aggregate weighted messages: [E, heads, head_dim] * [E, heads, 1]
        weighted_msg = h_src_edge * alpha_norm.unsqueeze(-1)
        out = torch.zeros(num_dst, self.heads, self.head_dim, device=x_dst.device)
        out.index_add_(0, dst_idx, weighted_msg)

        return out.view(num_dst, self.out_dim)


class HeteroGATLayer(nn.Module):
    """
    Heterogeneous GAT Layer performing message passing across all relations
    and aggregating incoming representations for each node type.
    """
    def __init__(self, node_dims: Dict[str, int], out_dim: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.out_dim = out_dim
        self.convs = nn.ModuleDict()

        # Define relations and create convolution heads
        relations = [
            ("process", "accesses", "file"),
            ("file", "accessed_by", "process"),
            ("process", "connects", "socket"),
            ("socket", "connected_by", "process"),
            ("socket", "transmits", "flow"),
            ("flow", "transmitted_by", "socket"),
        ]

        for src, rel, dst in relations:
            key = f"{src}__{rel}__{dst}"
            self.convs[key] = RelationGATConv(
                in_src=node_dims[src],
                in_dst=node_dims[dst],
                out_dim=out_dim,
                heads=heads,
                dropout=dropout
            )

        self.self_lins = nn.ModuleDict({
            ntype: nn.Linear(node_dims[ntype], out_dim) for ntype in node_dims
        })
        self.norms = nn.ModuleDict({
            ntype: nn.LayerNorm(out_dim) for ntype in node_dims
        })
        self.act = nn.GELU()

    def forward(
        self,
        node_feats: Dict[str, torch.Tensor],
        edge_indices: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        
        aggregated: Dict[str, List[torch.Tensor]] = {ntype: [] for ntype in node_feats}

        # Process each relation
        for (src, rel, dst), edge_idx in edge_indices.items():
            key = f"{src}__{rel}__{dst}"
            if key in self.convs and src in node_feats and dst in node_feats:
                msg = self.convs[key](node_feats[src], node_feats[dst], edge_idx)
                aggregated[dst].append(msg)

        # Combine incoming relation messages + self-projection
        out_feats: Dict[str, torch.Tensor] = {}
        for ntype, feat in node_feats.items():
            self_h = self.self_lins[ntype](feat)
            if aggregated[ntype]:
                # Mean aggregation over incoming relations + residual self connection
                rel_sum = torch.stack(aggregated[ntype], dim=0).mean(dim=0)
                combined = self_h + rel_sum
            else:
                combined = self_h
            
            normed = self.norms[ntype](combined)
            out_feats[ntype] = self.act(normed)

        return out_feats


class SFMGNNPlusModel(nn.Module):
    """
    SFM-GNN+ Multi-Modal Threat Detection Architecture.
    - Type-specific projection
    - 2-Layer Heterogeneous GAT
    - 64-dimensional Penultimate Layer for Decision Quality & FP suppression
    - Multi-Class Threat Classifier (10 classes: Benign + 9 Attacks)
    """
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        penultimate_dim: int = PENULTIMATE_DIM,
        num_classes: int = NUM_CLASSES,
        heads: int = NUM_HEADS,
        dropout: float = DROPOUT
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.penultimate_dim = penultimate_dim
        self.num_classes = num_classes

        # Initial projections for each node type
        self.input_projections = nn.ModuleDict({
            "process": nn.Linear(PROCESS_FEAT_DIM, hidden_dim),
            "file": nn.Linear(FILE_FEAT_DIM, hidden_dim),
            "socket": nn.Linear(SOCKET_FEAT_DIM, hidden_dim),
            "flow": nn.Sequential(
                nn.Linear(FLOW_FEAT_DIM, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        })

        layer1_dims = {k: hidden_dim for k in ["process", "file", "socket", "flow"]}
        self.gat1 = HeteroGATLayer(layer1_dims, hidden_dim, heads=heads, dropout=dropout)
        
        layer2_dims = {k: hidden_dim for k in ["process", "file", "socket", "flow"]}
        self.gat2 = HeteroGATLayer(layer2_dims, hidden_dim, heads=heads, dropout=dropout)

        # 64-dimensional Penultimate Latent Layer
        self.penultimate_layer = nn.Sequential(
            nn.Linear(hidden_dim, penultimate_dim),
            nn.LayerNorm(penultimate_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Final Classification Head (Logits)
        self.classifier = nn.Linear(penultimate_dim, num_classes)

    def forward(
        self,
        node_feats: Dict[str, torch.Tensor],
        edge_indices: Dict[Tuple[str, str, str], torch.Tensor],
        return_penultimate: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            logits: [num_flows, num_classes]
            penultimate: [num_flows, 64]
        """
        # 1. Project input features
        h = {}
        for ntype, feat in node_feats.items():
            if ntype in self.input_projections:
                h[ntype] = self.input_projections[ntype](feat)
            else:
                h[ntype] = feat

        # 2. HeteroGAT message passing
        h = self.gat1(h, edge_indices)
        h = self.gat2(h, edge_indices)

        # 3. Extract flow representations
        flow_h = h["flow"]

        # 4. Penultimate activation vector (64-dim)
        penultimate = self.penultimate_layer(flow_h)

        # 5. Classification logits
        logits = self.classifier(penultimate)

        if return_penultimate:
            return logits, penultimate
        return logits, None
