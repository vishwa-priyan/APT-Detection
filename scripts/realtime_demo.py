import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure root directory is on python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    ATTACK_CLASSES,
    CLASS_TO_IDX,
    NETFLOW_FEATURE_COLS,
    SAVED_MODELS_DIR,
)
from src.realtime.engine import RealtimeDetectionEngine
from src.data.netflow_loader import NetFlowLoader

def run_realtime_simulation(num_samples: int = 15):
    print("=" * 80)
    print(" SFM-GNN+: Real-Time Streaming Detection & SDN Remediation Simulation")
    print("=" * 80)

    # 1. Initialize Real-Time Engine
    print("[*] Initializing Real-Time Detection Engine with trained HeteroGAT weights...")
    engine = RealtimeDetectionEngine()
    print("[+] Engine initialized and calibrated successfully.")

    # 2. Extract realistic streaming test packets from dataset
    print("[*] Sampling streaming test flows across attack & benign categories...")
    loader = NetFlowLoader()
    df_test = loader.load_stratified_sample(samples_per_class=10, rare_class_boost=False)
    
    # Pick a curated sequence of diverse flows (e.g. Benign, Ransomware, Backdoor, Scanning, MITM)
    sample_df = df_test.sample(n=min(num_samples, len(df_test)), random_state=123).reset_index(drop=True)
    raw_flows = sample_df[NETFLOW_FEATURE_COLS].values
    ground_truth = sample_df["Attack"].tolist()

    metadata = []
    for i in range(len(sample_df)):
        metadata.append({
            "src_ip": f"192.168.1.{100 + (i % 20)}",
            "cid": 1024 + (i * 37) % 5000,
            "flow_id": f"flow-stream-{i+1:03d}"
        })

    print(f"\n[*] Streaming {len(sample_df)} live network flow events into detection pipeline...\n")
    print(f"{'FLOW ID':<16} | {'GROUND TRUTH':<12} | {'PREDICTED':<12} | {'CONF':<6} | {'STATUS':<26} | {'LATENCY':<8}")
    print("-" * 92)

    total_latency = 0.0
    confirmed_attacks = 0
    suppressed_fps = 0

    for idx in range(len(sample_df)):
        flow_vector = raw_flows[idx:idx+1]
        flow_meta = [metadata[idx]]

        t_start = time.perf_counter()
        results = engine.process_flow_batch(flow_vector, flow_meta)
        res = results[0]
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        total_latency += elapsed_ms

        gt = ground_truth[idx]
        pred = res["predicted_class_name"]
        conf = res["confidence"]
        status = res["verification"]["quality_status"]
        
        print(f"{metadata[idx]['flow_id']:<16} | {gt:<12} | {pred:<12} | {conf*100:5.1f}% | {status:<26} | {elapsed_ms:5.2f}ms")

        # If attack confirmed, show SDN remediation action and XAI explanation
        if res["verification"]["is_confirmed_attack"]:
            confirmed_attacks += 1
            mitigation = res["mitigation"]
            xai = res["explanation"]

            print(f"  [!] ALERT TRIGGERED: Confirmed {pred.upper()} Threat!")
            print(f"      -> SDN Action: {mitigation['ovs_cli_command']}")
            if mitigation.get("host_remediation"):
                print(f"      -> Host Containment: {mitigation['host_remediation']['command']} (PID {metadata[idx]['cid']})")
            if xai and xai.get("top_attributions"):
                top_features = ", ".join([f"{item['feature_name']} (score: {item['attribution_score']:.3f})" for item in xai["top_attributions"][:3]])
                print(f"      -> XAI Top Attributions: {top_features}")
            print()
        elif status == "FALSE_POSITIVE_SUPPRESSED":
            suppressed_fps += 1
            print("      -> [Penultimate Quality Check] False Positive suppressed! No network disruption triggered.")

    avg_latency = total_latency / len(sample_df)
    print("=" * 92)
    print(f"[+] Real-Time Streaming Evaluation Summary:")
    print(f"    - Total Flows Processed: {len(sample_df)}")
    print(f"    - Confirmed Attack Detections & SDN Mitigations: {confirmed_attacks}")
    print(f"    - Suppressed Borderline Alarms: {suppressed_fps}")
    print(f"    - Average End-to-End Latency: {avg_latency:.2f} ms / flow (High-throughput real-time capable)")
    print("=" * 92)

if __name__ == "__main__":
    run_realtime_simulation(num_samples=15)
