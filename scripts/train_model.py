import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Ensure root directory is on python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    SAVED_MODELS_DIR,
    ATTACK_CLASSES,
    CLASS_TO_IDX,
    LEARNING_RATE,
    WEIGHT_DECAY,
)
from src.data.netflow_loader import NetFlowLoader
from src.data.graph_builder import ThreatGraphBuilder
from src.models.hetero_gat import SFMGNNPlusModel
from src.models.contrastive_loss import MaskedFeatureRegularizer
from src.models.penultimate_verifier import PenultimateVerifier

def train():
    print("=" * 70)
    print(" SFM-GNN+: Training Secure Federated Heterogeneous GAT Model")
    print("=" * 70)

    device = torch.device("cpu")
    print(f"[*] Training Device: {device}")

    # 1. Load Stratified Sample from NF-ToN-IoT-V2
    print("[*] Ingesting and stratifying network flows across all 10 classes...")
    loader = NetFlowLoader()
    # Sample balanced set across all 10 classes
    df_sample = loader.load_stratified_sample(samples_per_class=4000, rare_class_boost=False)
    print(f"[+] Loaded {len(df_sample)} total flow records across classes:")
    print(df_sample["Attack"].value_counts())

    # Fit feature scaler
    print("[*] Normalizing flow features (41 dimensions)...")
    X_scaled, y_multi, y_binary = loader.fit_transform_features(df_sample)
    loader.save_scaler()
    print("[+] Feature scaler saved to saved_models/netflow_scaler.joblib")

    # Split train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_multi, test_size=0.2, random_state=42, stratify=y_multi
    )
    print(f"[+] Train set: {len(X_train)} samples | Validation set: {len(X_val)} samples")

    # Compute balanced class weights for CrossEntropyLoss
    class_counts = np.bincount(y_train, minlength=10)
    total_samples = len(y_train)
    weights = total_samples / (10.0 * np.maximum(class_counts, 1))
    weights = np.clip(weights, 0.2, 5.0)
    class_weights_t = torch.tensor(weights, dtype=torch.float32).to(device)

    # 2. Ingest CADETS Host Provenance Graph
    print("[*] Ingesting DARPA CADETS host provenance records (Processes, Files, Sockets, Events)...")
    graph_builder = ThreatGraphBuilder()
    graph_builder.load_host_context(max_records=30000)
    print(f"[+] CADETS Context: {len(graph_builder.cadets_parser.processes)} processes, "
          f"{len(graph_builder.cadets_parser.files)} files, "
          f"{len(graph_builder.cadets_parser.sockets)} sockets, "
          f"{len(graph_builder.cadets_parser.events)} causal events.")

    # 3. Construct Training & Validation Heterogeneous Threat Graphs
    print("[*] Building heterogeneous training graph...")
    train_graph = graph_builder.construct_fused_graph(
        flow_features=X_train,
        flow_labels=y_train,
        max_host_nodes=350
    )
    
    print("[*] Building heterogeneous validation graph...")
    val_graph = graph_builder.construct_fused_graph(
        flow_features=X_val,
        flow_labels=y_val,
        max_host_nodes=350
    )

    # 4. Initialize HeteroGAT Model & Optimizers
    model = SFMGNNPlusModel().to(device)
    masked_reg = MaskedFeatureRegularizer(feat_dim=41, hidden_dim=64).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(masked_reg.parameters()),
        lr=2e-3,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-4)

    # Prepare tensors for training
    train_node_feats = {k: v.to(device) for k, v in train_graph.node_features.items()}
    train_edge_indices = {k: v.to(device) for k, v in train_graph.edge_indices.items()}
    train_labels = train_graph.flow_labels.to(device)

    val_node_feats = {k: v.to(device) for k, v in val_graph.node_features.items()}
    val_edge_indices = {k: v.to(device) for k, v in val_graph.edge_indices.items()}
    val_labels = val_graph.flow_labels.to(device)

    print("\n[*] Starting HeteroGAT training loop with Masked Feature Regularization (MAGIC)...")
    num_epochs = 35
    best_f1 = 0.0
    best_penultimate = None

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        # Forward pass on train graph
        logits, penultimate = model(train_node_feats, train_edge_indices, return_penultimate=True)
        
        # Supervised Classification Loss
        cls_loss = criterion(logits, train_labels)

        # Masked Representation Regularization Loss
        recon_loss = masked_reg(train_node_feats["flow"], penultimate)
        total_loss = cls_loss + 0.1 * recon_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()

        # Validation Pass
        model.eval()
        with torch.no_grad():
            val_logits, val_penultimate = model(val_node_feats, val_edge_indices, return_penultimate=True)
            val_preds = torch.argmax(val_logits, dim=-1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_preds)
            val_f1 = f1_score(y_val, val_preds, average="weighted", zero_division=0)

        epoch_time = time.time() - t0

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(f" Epoch [{epoch:2d}/{num_epochs:2d}] | Loss: {total_loss.item():.4f} (Cls: {cls_loss.item():.4f}) | "
                  f"Val Acc: {val_acc * 100:.2f}% | Val F1: {val_f1:.4f} | Time: {epoch_time:.2f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_penultimate = val_penultimate.cpu().numpy()
            torch.save(model.state_dict(), SAVED_MODELS_DIR / "sfm_gnn_model.pt")

    print(f"\n[+] Best Validation Weighted F1: {best_f1:.4f}")
    print("[+] Saved best model weights to saved_models/sfm_gnn_model.pt")

    # 5. Final Evaluation & Classification Report
    print("\n" + "=" * 70)
    print(" Final Validation Classification Report Across 10 Classes:")
    print("=" * 70)
    model.load_state_dict(torch.load(SAVED_MODELS_DIR / "sfm_gnn_model.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        final_logits, final_pen = model(val_node_feats, val_edge_indices, return_penultimate=True)
        final_preds = torch.argmax(final_logits, dim=-1).cpu().numpy()
    
    unique_classes_in_val = sorted(list(set(y_val) | set(final_preds)))
    target_names = [ATTACK_CLASSES[i] for i in unique_classes_in_val]
    report = classification_report(y_val, final_preds, labels=unique_classes_in_val, target_names=target_names, digits=4, zero_division=0)
    print(report)

    # 6. Calibrate Penultimate Verifier (Algorithm 2)
    print("[*] Calibrating Penultimate Decision Verifier (64-dim cluster centroids & radii)...")
    verifier = PenultimateVerifier()
    verifier.calibrate(best_penultimate, y_val)
    verifier.save()
    print("[+] Penultimate Verifier calibrated and saved to saved_models/penultimate_verifier.joblib")

    print("\n[OK] Model development, training, and penultimate calibration completed successfully!")

if __name__ == "__main__":
    train()
