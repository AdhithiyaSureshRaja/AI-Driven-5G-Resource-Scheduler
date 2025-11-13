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