## Author
Adhithiya Suresh Raja  
adhi200328@gmail.com

```
```

---

## requirements.txt

```text
numpy
pandas
matplotlib
torch
scapy
tqdm
seaborn

# Optional: for dashboard
streamlit
```

---

## LICENSE (MIT)

```text
MIT License

Copyright (c) 2025 Adhithiya Suresh Raja

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## src/utils.py

```python
"""Utility functions: normalization, reward calc, logging helpers."""
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
```

---

## src/data_extraction.py

```python
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
```

---

## src/train_rl_model.py

```python
"""Train a simple Deep Q-Network (DQN) to allocate PRBs across UEs using simulated telemetry.
This is a lightweight educational implementation (not production-grade)."""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import pandas as pd
import os
from utils import ensure_dir, compute_reward


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class SimpleDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        batch = [self.buf[i] for i in idx]
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buf)


class EnvSimulator:
    """Very simple environment: state is vector of CQIs for N UEs. Action: distribution of PRBs.
    Reward: weighted throughput/latency/fairness. This is intentionally simple for training demos."""

    def __init__(self, n_ues=4, max_prbs=20):
        self.n = n_ues
        self.max_prbs = max_prbs

    def reset(self):
        # state: CQI values
        self.state = np.random.randint(1, 16, size=(self.n,)).astype(np.float32)
        return self.state.copy()

    def step(self, action):
        # action: an integer in [0, max_prbs] representing PRBs assigned to UE idx = action_idx
        prb_allocation = np.zeros(self.n, dtype=np.int32)
        # decode action as allocation vector: here action is index of UE to give one PRB (simple)
        chosen_ue = action
        prb_allocation[chosen_ue] = 1
        throughput = (self.state * (1 + prb_allocation)).sum() * random.uniform(3.0, 6.0)
        latency = max(1.0, 100.0 / (throughput + 1.0))
        fairness = 1.0 - np.std(self.state) / 15.0
        reward = compute_reward(throughput, latency, fairness)
        # next state random walk
        self.state = np.clip(self.state + np.random.randint(-1, 2, size=(self.n,)), 1, 15)
        done = False
        return self.state.copy(), reward, done, {"throughput": throughput, "latency": latency, "fairness": fairness}


def train(args):
    n_ues = args.ues
    env = EnvSimulator(n_ues=n_ues)
    input_dim = n_ues
    output_dim = n_ues  # choose which UE to give PRB (very simplified)

    dqn = SimpleDQN(input_dim, output_dim)
    optimizer = optim.Adam(dqn.parameters(), lr=1e-3)
    buffer = ReplayBuffer(20000)

    eps = 1.0
    eps_min = 0.05
    eps_decay = 0.995

    for ep in range(args.episodes):
        state = env.reset()
        total_r = 0.0
        for step in range(args.steps):
            if random.random() < eps:
                action = random.randrange(output_dim)
            else:
                with torch.no_grad():
                    qvals = dqn(torch.tensor(state, dtype=torch.float32))
                    action = int(torch.argmax(qvals).item())
            next_state, reward, done, info = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            total_r += reward

            if len(buffer) > 128:
                batch = buffer.sample(64)
                states = torch.tensor(batch.state, dtype=torch.float32)
                actions = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1)
                rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
                next_states = torch.tensor(batch.next_state, dtype=torch.float32)

                q_values = dqn(states).gather(1, actions)
                next_q = dqn(next_states).max(1)[0].detach().unsqueeze(1)
                target = rewards + 0.99 * next_q
                loss = torch.nn.functional.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

        eps = max(eps_min, eps * eps_decay)
        if ep % 10 == 0:
            print(f"Episode {ep} avg reward {total_r / args.steps:.4f} eps {eps:.3f}")

    ensure_dir("results")
    torch.save(dqn.state_dict(), "results/dqn_scheduler.pt")
    print("Training complete. Model saved to results/dqn_scheduler.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--ues", type=int, default=4)
    args = parser.parse_args()
    train(args)
```

---

## src/ai_scheduler.py

```python
"""A simple runtime that loads the trained model and acts as a scheduler in simulate or demo mode.
It shows how the trained model could be queried to pick scheduling actions.
"""
import argparse
import torch
import numpy as np
from utils import ensure_dir


class SimpleDQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def run_simulator(model_path: str, steps: int = 100, n_ues: int = 4):
    model = SimpleDQN(n_ues, n_ues)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    state = np.random.randint(1, 16, size=(n_ues,)).astype(np.float32)
    for t in range(steps):
        with torch.no_grad():
            q = model(torch.tensor(state, dtype=torch.float32))
            action = int(torch.argmax(q).item())
        # Apply action (very simple): allocate one PRB to chosen UE
        print(f"t={t} state={state.tolist()} action=UE_{action}")
        # random walk for next state
        state = np.clip(state + np.random.randint(-1, 2, size=(n_ues,)), 1, 15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["simulate", "live"], default="simulate")
    parser.add_argument("--model", default="results/dqn_scheduler.pt")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "simulate":
        run_simulator(args.model, steps=args.steps)
    else:
        print("Live mode not implemented in this template. Connect to srsRAN hooks to integrate.")
```

---

## src/evaluate_performance.py

```python
"""Evaluate trained agent vs baseline schedulers by running multiple simulated episodes and
collecting metrics such as throughput/latency/fairness. Produces a CSV and a comparison plot.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
import torch
from train_rl_model import EnvSimulator, SimpleDQN
from utils import ensure_dir, save_metrics


def rr_scheduler(state):
    # Round robin: pick next UE in increasing order (very basic stub)
    return np.argmin(state)  # naive: serve worst-state UE


def pf_scheduler(state):
    # Proportional fair: serve highest CQI * fairness approximation
    return int(np.argmax(state))


def evaluate(model_path: str, episodes: int = 100, steps: int = 50, n_ues: int = 4):
    env = EnvSimulator(n_ues=n_ues)
    ensure_dir("results")

    # load model
    model = SimpleDQN(n_ues, n_ues)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    rows = []
    for ep in trange(episodes):
        state = env.reset()
        for s in range(steps):
            # AI action
            with torch.no_grad():
                q = model(torch.tensor(state, dtype=torch.float32))
                ai_action = int(torch.argmax(q).item())
            ns, r_ai, _, info_ai = env.step(ai_action)

            # RR baseline
            rr_action = rr_scheduler(state)
            _, r_rr, _, info_rr = env.step(rr_action)

            # PF baseline
            pf_action = pf_scheduler(state)
            _, r_pf, _, info_pf = env.step(pf_action)

            rows.append({
                'episode': ep,
                'step': s,
                'ai_throughput': info_ai['throughput'],
                'ai_latency': info_ai['latency'],
                'ai_reward': r_ai,
                'rr_throughput': info_rr['throughput'],
                'pf_throughput': info_pf['throughput'],
            })
            state = ns

    df = pd.DataFrame(rows)
    save_metrics(df, "results/eval_metrics.csv")

    # aggregate and plot
    agg = df.groupby('step').mean()
    plt.figure(figsize=(8, 5))
    plt.plot(agg['ai_throughput'], label='AI')
    plt.plot(agg['rr_throughput'], label='RR')
    plt.plot(agg['pf_throughput'], label='PF')
    plt.xlabel('Step')
    plt.ylabel('Throughput (arb units)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/comparison_plot.png')
    print('Evaluation complete. Results saved to results/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='results/dqn_scheduler.pt')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--steps', type=int, default=40)
    args = parser.parse_args()
    evaluate(args.model, episodes=args.episodes, steps=args.steps)
```

---

## srsran_setup/install_srsran.sh

```bash
#!/usr/bin/env bash
set -e

# Basic install script for srsRAN (template). Modify based on your distro and requirements.
# This script only provides a helper — refer to official srsRAN docs for specifics.

sudo apt update && sudo apt upgrade -y
sudo apt install -y git build-essential cmake libboost-all-dev libsctp-dev libfftw3-dev libmbedtls-dev libmbedtls12 libasio-dev libzmq3-dev libssl-dev python3-pip

# Clone srsRAN
if [ ! -d "srsRAN" ]; then
  git clone https://github.com/srsran/srsRAN.git
  cd srsRAN
  mkdir build && cd build
  cmake ..
  make -j$(nproc)
  sudo make install
  cd ../..
else
  echo "srsRAN folder exists. Skipping clone."
fi

# Python deps
pip3 install --user numpy pandas matplotlib

echo "Install script finished. Review srsRAN docs for additional radio/hardware setup steps."
```

---

## srsran_setup/gnb.conf

```ini
# Placeholder gNB config. Adapt to your srsRAN version and hardware.
[rf]
# radio configuration
frequency=3600000000
# sample_rate etc.

[phy]
# physical layer settings

[mac]
# MAC settings


# NOTE: This is a template. Use srsRAN's example configs as your baseline.
```

---

## srsran_setup/ue.conf

```ini
# Placeholder UE config. Adapt to your srsRAN version and hardware.
[rf]
# radio configuration
frequency=3600000000

[phy]

[mac]

# NOTE: Template only.
```

---

## results/

Add your generated `evaluation_plot.png`, `metrics.csv`, and any exported logs here after running scripts.

---

# End of textdoc

