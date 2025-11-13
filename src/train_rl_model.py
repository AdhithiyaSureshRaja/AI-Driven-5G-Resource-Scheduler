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