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
