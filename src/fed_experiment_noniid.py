"""
Federated Learning Experiment — Non-IID Regime Test (v3.0)
===========================================================
Pre-registration: prereg/PRE_REGISTRATION_v3.md

Hypothesis: In a non-IID setup (each client sees only 2 digit classes),
the diversity penalty γ is much larger, so an interior optimum ρ* ∈ (0,1)
should re-emerge — rehabilitating P1 under a scoped form.

Architecture & FedAvg loop match v2 (fed_experiment.py) exactly:
  - Same SmallCNN (conv1 → conv2 → fc1 → fc2, ~50k params)
  - Same global-model pattern (each round: selected clients fork from
    global model, train locally, average back)
  - Same MNIST normalization (mean=0.1307, std=0.3081)
  - 10 rounds, 3 seeds, 1000 samples per client

The ONLY difference from v2 is the data split: each client sees 2 digit
classes instead of a random IID subset.

Run: python fed_experiment_noniid.py
Time: ~5–10 minutes on CPU
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import json
import time
import os

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ────────────────────────────────────────────────────────────
# Same model as v2 (SmallCNN, ~50k params)
# ────────────────────────────────────────────────────────────
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ────────────────────────────────────────────────────────────
# Non-IID Split — each client gets exactly 2 digit classes
# ────────────────────────────────────────────────────────────
def create_noniid_splits(dataset, num_clients=10, samples_per_client=1000):
    """
    Assign 2 digits per client so clients have maximally different
    data distributions — this is where the diversity penalty γ is large.
    """
    targets = np.array(dataset.targets)
    digit_groups = [
        [0, 1], [2, 3], [4, 5], [6, 7], [8, 9],
        [0, 2], [1, 3], [4, 6], [5, 7], [8, 0],
    ]
    client_subsets = []
    for group in digit_groups[:num_clients]:
        idx = np.where(np.isin(targets, group))[0]
        np.random.shuffle(idx)
        idx = idx[:samples_per_client]
        client_subsets.append(Subset(dataset, idx))
    return client_subsets


# ────────────────────────────────────────────────────────────
# Local training (same as v2)
# ────────────────────────────────────────────────────────────
def local_train(model, loader, epochs=1, lr=0.01):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def average_states(states):
    """Average model weights from a subset of clients."""
    avg = {k: torch.zeros_like(v) for k, v in states[0].items()}
    for s in states:
        for k in avg:
            avg[k] += s[k] / len(states)
    return avg


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


# ────────────────────────────────────────────────────────────
# FedAvg run — SAME pattern as v2 (global model, fork, train, avg)
# ────────────────────────────────────────────────────────────
def federated_run(client_loaders, test_loader, rho, rounds=10, sample_seed=0):
    """
    Standard FedAvg: one global model. Each round, ρ fraction of clients
    fork from global, train locally on their non-IID data, and average back.
    """
    np.random.seed(sample_seed)
    global_model = SmallCNN().to(DEVICE)
    n_clients = len(client_loaders)
    n_participants = max(1, int(round(rho * n_clients)))
    total_comm = 0

    for r in range(rounds):
        chosen = np.random.choice(n_clients, n_participants, replace=False)
        local_states = []
        for c in chosen:
            local = SmallCNN().to(DEVICE)
            local.load_state_dict(global_model.state_dict())
            state = local_train(local, client_loaders[c], epochs=1)
            local_states.append(state)
        global_model.load_state_dict(average_states(local_states))
        total_comm += n_participants

    acc = evaluate(global_model, test_loader)
    return acc, total_comm


# ────────────────────────────────────────────────────────────
# Main experiment
# ────────────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("NON-IID FEDERATED EXPERIMENT (v3.0)")
    print("Testing P1-v3, P2-v3, P3-v3 from PRE_REGISTRATION_v3.md")
    print("Same model & FedAvg as v2; ONLY change is non-IID client data")
    print("=" * 64)

    # Same normalization as v2
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test, batch_size=256, shuffle=False)

    rhos = [0.1, 0.3, 0.5, 0.7, 1.0]
    n_seeds = 3

    # Build non-IID client splits
    client_subsets = create_noniid_splits(train, num_clients=10,
                                          samples_per_client=1000)
    client_loaders = [DataLoader(c, batch_size=32, shuffle=True)
                      for c in client_subsets]

    # Log the split for transparency
    print("\nClient digit assignments:")
    digit_groups = [
        [0, 1], [2, 3], [4, 5], [6, 7], [8, 9],
        [0, 2], [1, 3], [4, 6], [5, 7], [8, 0],
    ]
    for i, group in enumerate(digit_groups[:10]):
        print(f"  Client {i}: digits {group}")

    results = []
    print(f"\n{'─' * 64}")
    print(f"  Running {len(rhos)} ρ values × {n_seeds} seeds = "
          f"{len(rhos) * n_seeds} runs")
    print(f"{'─' * 64}")

    for rho in rhos:
        t0 = time.time()
        accs, comms = [], []
        for s in range(n_seeds):
            acc, comm = federated_run(client_loaders, test_loader,
                                      rho, rounds=10, sample_seed=s)
            accs.append(acc)
            comms.append(comm)
        acc_mean = float(np.mean(accs))
        acc_std  = float(np.std(accs))
        comm_mean = float(np.mean(comms))
        print(f"  ρ={rho:.1f}  acc={acc_mean:.4f}±{acc_std:.4f}  "
              f"comm={comm_mean:.0f}  time={time.time()-t0:.1f}s")
        results.append({
            "rho": rho,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "comm": comm_mean,
        })

    # ── Save results ────────────────────────────────────────
    os.makedirs("../results", exist_ok=True)
    out = {
        "experiment": "non-IID regime test v3.0",
        "date": "2026-04-25",
        "setup": {
            "clients": 10,
            "digits_per_client": 2,
            "rounds": 10,
            "seeds": n_seeds,
            "samples_per_client": 1000,
            "model": "SmallCNN (same as v2)",
            "normalization": "mean=0.1307, std=0.3081",
            "label_noise": 0.0,
        },
        "results": results,
    }
    with open("../results/noniid_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n✅ Results saved: ../results/noniid_results.json")

    # ── Verdict ─────────────────────────────────────────────
    accs_all = [r["acc_mean"] for r in results]
    peak_idx = int(np.argmax(accs_all))
    peak_rho = results[peak_idx]["rho"]
    is_interior = 0 < peak_idx < len(results) - 1

    print("\n" + "=" * 64)
    print("VERDICT (Pre-Registered)")
    print("=" * 64)
    print(f"  Peak ρ* = {peak_rho}")
    if is_interior:
        print(f"  ✅ P1-v3: PASS — interior optimum at ρ*={peak_rho}")
    else:
        print(f"  ❌ P1-v3: FAIL — boundary peak at ρ*={peak_rho}")
    print()

    # ── Plot ────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        rs = [r["rho"] for r in results]
        accs = [r["acc_mean"] for r in results]
        stds = [r["acc_std"] for r in results]
        cs = [r["comm"] for r in results]

        axes[0].errorbar(rs, accs, yerr=stds, marker="o",
                         label="non-IID (2 classes/client)", capsize=3,
                         color="#9C27B0", linewidth=2)
        axes[0].scatter([peak_rho], [accs[peak_idx]], s=120, zorder=5,
                        color="#9C27B0", edgecolor="black", linewidth=1.5)
        axes[0].set_xlabel("Sync density ρ")
        axes[0].set_ylabel("Test accuracy")
        axes[0].set_title("v3 Non-IID: Accuracy vs. Sync Density")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(rs, cs, marker="s", color="#9C27B0", linewidth=2)
        axes[1].set_xlabel("Sync density ρ")
        axes[1].set_ylabel("Communication cost (proxy)")
        axes[1].set_title("v3 Non-IID: Communication Cost")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("../results/noniid_results.png", dpi=120)
        print("✅ Plot saved: ../results/noniid_results.png")
    except ImportError:
        print("⚠️  matplotlib not installed — skipping plot")


if __name__ == "__main__":
    main()
