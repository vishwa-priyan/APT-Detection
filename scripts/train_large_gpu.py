import os
import sys
import time
from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Ensure root directory is on python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    SAVED_MODELS_DIR,
    NETFLOW_PARQUET_PATH,
    NETFLOW_FEATURE_COLS,
    ATTACK_CLASSES,
    CLASS_TO_IDX,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_CLASSES,
)
from src.data.netflow_loader import NetFlowLoader
from src.data.graph_builder import ThreatGraphBuilder
from src.models.hetero_gat import SFMGNNPlusModel
from src.models.contrastive_loss import MaskedFeatureRegularizer
from src.models.penultimate_verifier import PenultimateVerifier

class FlowTensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_25_percent_dataset(parquet_path: str, target_total: int = 3283970) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads 25% of the dataset (~3.28 million flows) from the Parquet dataset
    using PyArrow chunk scanning with stratified class balance.
    """
    print(f"[*] Scanning Parquet dataset to load ~{target_total:,} flows (25% of total 13.1M)...")
    dataset = ds.dataset(parquet_path, format="parquet")
    needed_cols = NETFLOW_FEATURE_COLS + ["Attack"]
    
    # We allocate proportional target quotas per class to ensure rare classes are completely captured
    # rare classes: capture 100% of ransomware (3,357), 100% of mitm (7,723), 100% of backdoor (16,259)
    # The remaining ~3.25 million rows are distributed evenly across the other 7 classes (~465,000 per class)
    common_quota = int((target_total - 30000) / 7)
    class_targets = {}
    for cls in ATTACK_CLASSES:
        if cls == "ransomware":
            class_targets[cls] = 3357
        elif cls == "mitm":
            class_targets[cls] = 7723
        elif cls == "backdoor":
            class_targets[cls] = 16259
        else:
            class_targets[cls] = common_quota

    collected = {cls: [] for cls in ATTACK_CLASSES}
    collected_counts = {cls: 0 for cls in ATTACK_CLASSES}

    scanner = dataset.scanner(columns=needed_cols, batch_size=250000)
    total_gathered = 0

    t0 = time.time()
    for batch in scanner.to_batches():
        chunk = batch.to_pandas()
        for cls in ATTACK_CLASSES:
            target = class_targets[cls]
            current = collected_counts[cls]
            if current < target:
                match = chunk[chunk["Attack"] == cls]
                if len(match) > 0:
                    needed = target - current
                    selected = match.head(needed)
                    # Convert to float32 features immediately to optimize RAM
                    raw_x = np.clip(selected[NETFLOW_FEATURE_COLS].values, -1e12, 1e12).astype(np.float32)
                    labels = np.full(len(selected), CLASS_TO_IDX[cls], dtype=np.int64)
                    collected[cls].append((raw_x, labels))
                    collected_counts[cls] += len(selected)
                    total_gathered += len(selected)

        # Check if all quotas reached
        done = all(collected_counts[c] >= class_targets[c] for c in ATTACK_CLASSES)
        if done:
            break

    elapsed = time.time() - t0
    print(f"[+] Loaded {total_gathered:,} flows in {elapsed:.1f}s. Class breakdown:")
    for cls in ATTACK_CLASSES:
        print(f"    - {cls:<12}: {collected_counts[cls]:,} flows")

    # Merge arrays efficiently
    all_x = []
    all_y = []
    for cls in ATTACK_CLASSES:
        for x_part, y_part in collected[cls]:
            all_x.append(x_part)
            all_y.append(y_part)

    X_all = np.vstack(all_x)
    y_all = np.concatenate(all_y)

    # Shuffle
    perm = np.random.permutation(len(X_all))
    return X_all[perm], y_all[perm]

def train_large():
    print("=" * 80)
    print(" SFM-GNN+: Large-Scale 25% Dataset Training on NVIDIA RTX 3050 GPU")
    print("=" * 80)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Compute Device: {device}")
    if device.type == "cuda":
        print(f"[+] Active GPU: {torch.cuda.get_device_name(0)}")
        print(f"[+] VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        torch.backends.cudnn.benchmark = True
    else:
        print("[-] Notice: CUDA is not active yet. Falling back to CPU.")

    # 1. Ingest CADETS Host Provenance
    print("[*] Ingesting DARPA CADETS host provenance context (50,000 records)...")
    graph_builder = ThreatGraphBuilder()
    graph_builder.load_host_context(max_records=50000)
    print(f"[+] CADETS Context: {len(graph_builder.cadets_parser.processes)} processes, "
          f"{len(graph_builder.cadets_parser.files)} files, "
          f"{len(graph_builder.cadets_parser.sockets)} sockets, "
          f"{len(graph_builder.cadets_parser.events)} causal events.")

    # 2. Ingest 25% of NF-ToN-IoT-V2
    X_raw, y_raw = load_25_percent_dataset(str(NETFLOW_PARQUET_PATH), target_total=3283970)

    # 3. Fit or load feature scaler
    print("[*] Normalizing 3.28 Million flow features with StandardScaler...")
    loader = NetFlowLoader()
    # Fit scaler on a representative 100K chunk to save memory, then transform in-place
    loader.scaler.fit(X_raw[:100000])
    loader.is_fitted = True
    loader.save_scaler()

    # Transform in chunks to conserve RAM
    chunk_sz = 200000
    for i in range(0, len(X_raw), chunk_sz):
        X_raw[i:i+chunk_sz] = loader.scaler.transform(X_raw[i:i+chunk_sz])
    print("[+] Feature normalization complete and scaler saved.")

    # Train / Val split (90% train, 10% validation = ~328,000 val samples)
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=0.10, random_state=42, stratify=y_raw
    )
    print(f"[+] Train Partition: {len(X_train):,} samples | Validation Partition: {len(X_val):,} samples")

    # DataLoader
    batch_size = 4096 if device.type == "cuda" else 2048
    train_dataset = FlowTensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"))

    # Compute balanced class weights
    class_counts = np.bincount(y_train, minlength=10)
    total_samples = len(y_train)
    weights = total_samples / (10.0 * np.maximum(class_counts, 1))
    weights = np.clip(weights, 0.2, 5.0)
    class_weights_t = torch.tensor(weights, dtype=torch.float32).to(device)

    # Build base host graph template on device
    host_dummy_flows = np.zeros((10, 41), dtype=np.float32)
    template_graph = graph_builder.construct_fused_graph(
        flow_features=host_dummy_flows,
        flow_labels=np.zeros(10, dtype=np.int64),
        max_host_nodes=200
    )
    
    host_node_feats = {
        k: v.to(device) for k, v in template_graph.node_features.items() if k != "flow"
    }

    # Model Initialization
    model = SFMGNNPlusModel().to(device)
    masked_reg = MaskedFeatureRegularizer(feat_dim=41, hidden_dim=64).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(masked_reg.parameters()),
        lr=2e-3,
        weight_decay=1e-4
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-4)

    print("\n[*] Starting 25% Large-Scale Training across 12 Epochs on RTX 3050...")
    num_epochs = 12
    best_val_f1 = 0.0

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss_accum = 0.0
        num_batches = 0

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            cur_bs = batch_x.size(0)

            # Combine host node feats with current batch flow feats
            curr_feats = {k: v for k, v in host_node_feats.items()}
            curr_feats["flow"] = batch_x

            # Construct dynamic socket-flow edges for current batch
            num_sockets = host_node_feats["socket"].size(0)
            s_idx = torch.arange(cur_bs, device=device) % num_sockets
            f_idx = torch.arange(cur_bs, device=device)
            
            curr_edges = {
                ("socket", "transmits", "flow"): torch.stack([s_idx, f_idx]),
                ("flow", "transmitted_by", "socket"): torch.stack([f_idx, s_idx]),
                ("process", "accesses", "file"): template_graph.edge_indices[("process", "accesses", "file")].to(device),
                ("file", "accessed_by", "process"): template_graph.edge_indices[("file", "accessed_by", "process")].to(device),
                ("process", "connects", "socket"): template_graph.edge_indices[("process", "connects", "socket")].to(device),
                ("socket", "connected_by", "process"): template_graph.edge_indices[("socket", "connected_by", "process")].to(device),
            }

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, penultimate = model(curr_feats, curr_edges, return_penultimate=True)
                cls_loss = criterion(logits, batch_y)
                recon_loss = masked_reg(batch_x, penultimate)
                loss = cls_loss + 0.05 * recon_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss_accum += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 200 == 0:
                print(f"  [Epoch {epoch:02d} | Step {batch_idx+1:04d}/{len(train_loader):04d}] Batch Loss: {loss.item():.4f}")

        scheduler.step()
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss_accum / max(num_batches, 1)

        # Quick validation on 40,000 sample subset
        model.eval()
        val_sub_x = torch.from_numpy(X_val[:40000]).float().to(device)
        val_sub_y = y_val[:40000]
        with torch.no_grad():
            curr_feats = {k: v for k, v in host_node_feats.items()}
            curr_feats["flow"] = val_sub_x
            val_bs = val_sub_x.size(0)
            s_idx = torch.arange(val_bs, device=device) % num_sockets
            f_idx = torch.arange(val_bs, device=device)
            val_edges = {
                ("socket", "transmits", "flow"): torch.stack([s_idx, f_idx]),
                ("flow", "transmitted_by", "socket"): torch.stack([f_idx, s_idx]),
                ("process", "accesses", "file"): template_graph.edge_indices[("process", "accesses", "file")].to(device),
                ("file", "accessed_by", "process"): template_graph.edge_indices[("file", "accessed_by", "process")].to(device),
                ("process", "connects", "socket"): template_graph.edge_indices[("process", "connects", "socket")].to(device),
                ("socket", "connected_by", "process"): template_graph.edge_indices[("socket", "connected_by", "process")].to(device),
            }
            val_logits, val_pen = model(curr_feats, val_edges, return_penultimate=True)
            val_preds = torch.argmax(val_logits, dim=-1).cpu().numpy()
            val_acc = accuracy_score(val_sub_y, val_preds)
            val_f1 = f1_score(val_sub_y, val_preds, average="weighted", zero_division=0)

        print(f"==> Epoch [{epoch:02d}/{num_epochs:02d}] Finished in {epoch_time:.1f}s | "
              f"Avg Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), SAVED_MODELS_DIR / "sfm_gnn_model.pt")

    print(f"\n[+] Training Complete! Best Validation F1: {best_val_f1:.4f}")
    
    # Calibrate Penultimate Verifier with final representations
    print("[*] Calibrating Penultimate Decision Verifier (64-dim)...")
    verifier = PenultimateVerifier()
    verifier.calibrate(val_pen.cpu().numpy(), val_sub_y)
    verifier.save()
    print("[+] Penultimate Verifier calibrated and saved.")

    # Final Classification Report on full Validation Set
    print("\n" + "=" * 80)
    print(" Final Classification Report on Validation Set Across All 10 Classes:")
    print("=" * 80)
    report = classification_report(val_sub_y, val_preds, target_names=ATTACK_CLASSES, digits=4, zero_division=0)
    print(report)
    print("[OK] Large-Scale 25% training successfully completed and saved.")

if __name__ == "__main__":
    train_large()
