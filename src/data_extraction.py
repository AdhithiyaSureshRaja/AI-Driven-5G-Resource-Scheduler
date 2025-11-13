"""Extract telemetry from srsRAN logs or simulate telemetry for development.

This script supports two modes:
 - parse: given a folder with srsRAN logs, parse CQI/PRB/throughput lines
 - simulate: generate synthetic telemetry to train/evaluate models without real hardware

Usage:
 python3 data_extraction.py --simulate
 python3 data_extraction.py --parse /path/to/logs/
"""

import argparse
import random
import csv
import os
from datetime import datetime
from utils import ensure_dir


def simulate_telemetry(num_steps=100, num_ues=4, out_csv="results/metrics_sim.csv"):
    ensure_dir(os.path.dirname(out_csv))
    fields = ["time", "ue_id", "cqi", "prb_used", "throughput_kbps", "latency_ms"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in range(num_steps):
            for ue in range(num_ues):
                cqi = max(1, min(15, int(random.gauss(8 + ue, 2))))
                prb = random.randint(1, 10)
                throughput = max(50, int(cqi * prb * random.uniform(4.0, 7.0)))
                latency = max(1, int(random.expovariate(1/30.0)))
                w.writerow({
                    "time": datetime.utcnow().isoformat(),
                    "ue_id": ue,
                    "cqi": cqi,
                    "prb_used": prb,
                    "throughput_kbps": throughput,
                    "latency_ms": latency,
                })
    print(f"Simulated telemetry saved to {out_csv}")


def parse_srsran_logs(log_dir: str, out_csv="results/metrics_parsed.csv"):
    # Placeholder parser: srsRAN logs vary by version. This provides an example template.
    ensure_dir(os.path.dirname(out_csv))
    fields = ["time", "ue_id", "cqi", "prb_used", "throughput_kbps", "latency_ms"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        # Walk logs and pull telemetry lines — users should adapt regexes to their version of srsRAN
        for root, _, files in os.walk(log_dir):
            for file in files:
                path = os.path.join(root, file)
                with open(path, "r", errors="ignore") as fh:
                    for line in fh:
                        if "CQI" in line or "PRB" in line:
                            # naive parsing example — adapt to actual log format
                            parts = line.strip().split()
                            # This is a placeholder: extract sensible defaults
                            time = datetime.utcnow().isoformat()
                            ue = 0
                            cqi = random.randint(1, 15)
                            prb = random.randint(1, 10)
                            throughput = random.randint(50, 2000)
                            latency = random.randint(1, 200)
                            w.writerow({
                                "time": time,
                                "ue_id": ue,
                                "cqi": cqi,
                                "prb_used": prb,
                                "throughput_kbps": throughput,
                                "latency_ms": latency,
                            })
    print(f"Parsed telemetry saved to {out_csv} (template parser)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true", help="Generate simulated telemetry")
    parser.add_argument("--parse", type=str, help="Path to srsRAN logs folder to parse")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--ues", type=int, default=4)
    args = parser.parse_args()

    if args.simulate:
        simulate_telemetry(num_steps=args.steps, num_ues=args.ues)
    elif args.parse:
        parse_srsran_logs(args.parse)
    else:
        parser.print_help()