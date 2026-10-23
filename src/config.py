import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
CADETS_DIR = DATASET_DIR / "DARPA TC E3 – CADETS Dataset" / "ta1-cadets-e3-official-1.json"
NETFLOW_PARQUET_PATH = DATASET_DIR / "Network Traffic Datasets (NetFlow & Packets)" / "NF-ToN-IoT-V2.parquet"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Attack Classes Mapping (10 classes: 0 = Benign, 1..9 = Attacks)
ATTACK_CLASSES = [
    "Benign",
    "scanning",
    "xss",
    "ddos",
    "password",
    "injection",
    "dos",
    "backdoor",
    "mitm",
    "ransomware",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(ATTACK_CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(ATTACK_CLASSES)}
NUM_CLASSES = len(ATTACK_CLASSES)

# Feature Names from NF-ToN-IoT-V2 (41 features)
NETFLOW_FEATURE_COLS = [
    'L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'IN_PKTS',
    'OUT_BYTES', 'OUT_PKTS', 'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
    'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT', 'MIN_TTL', 'MAX_TTL',
    'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN',
    'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES', 'RETRANSMITTED_IN_BYTES',
    'RETRANSMITTED_IN_PKTS', 'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS',
    'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT', 'NUM_PKTS_UP_TO_128_BYTES',
    'NUM_PKTS_128_TO_256_BYTES', 'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES',
    'NUM_PKTS_1024_TO_1514_BYTES', 'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT', 'ICMP_TYPE',
    'ICMP_IPV4_TYPE', 'DNS_QUERY_ID', 'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER', 'FTP_COMMAND_RET_CODE'
]
NUM_FLOW_FEATURES = len(NETFLOW_FEATURE_COLS)

# Dimensions for Heterogeneous Nodes
PROCESS_FEAT_DIM = 16
FILE_FEAT_DIM = 8
SOCKET_FEAT_DIM = 8
FLOW_FEAT_DIM = NUM_FLOW_FEATURES  # 41

# Model Hyperparameters
HIDDEN_DIM = 64
PENULTIMATE_DIM = 64
NUM_HEADS = 4
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Real-Time & SDN Settings
SDN_CONTROLLER_URL = "http://127.0.0.1:8080"
DEFAULT_PENULTIMATE_THRESHOLD = 0.85
