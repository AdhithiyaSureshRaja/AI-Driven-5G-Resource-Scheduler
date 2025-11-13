import numpy as np
import pandas as pd
import os
import json
from typing import Dict, Any


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def save_metrics(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def compute_reward(throughput: float, latency: float, fairness: float, w_t=0.6, w_l=0.3, w_f=0.1) -> float:
    """Simple scalar reward — higher throughput good, lower latency good, higher fairness good."""
    # Normalize assumed ranges (user to adapt)
    t = throughput / 1000.0  # example normalization
    l = 1.0 / (1.0 + latency)
    f = fairness
    return w_t * t + w_l * l + w_f * f


def summarize_metrics(metrics_list: list) -> Dict[str, float]:
    df = pd.DataFrame(metrics_list)
    return df.mean().to_dict()