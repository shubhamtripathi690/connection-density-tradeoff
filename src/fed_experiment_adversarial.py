"""
Federated Learning Experiment — Adversarial Clients (v4.0)
===========================================================
Pre-registration: prereg/PRE_REGISTRATION_v4.md

Hypothesis: When 30% of clients have flipped labels (y → 9−y),
their gradients actively conflict with honest clients. Averaging
adversarial updates into the global model should HARM accuracy,
creating an interior optimum where partial sync beats full sync.

Architecture & FedAvg loop identical to v2/v3:
  - SmallCNN (conv1 → conv2 → fc1 → fc2, ~50k params)
  - Global-model FedAvg: fork → train → average → broadcast
  - MNIST normalization (mean=0.1307, std=0.3081)
  - 10 rounds, 3 seeds, 1000 samples per client

The ONLY change: 3 of 10 clients have ALL labels flipped (y → 9−y).

Run: python fed_experiment_adversarial.py
Time: ~5–10 minutes on CPU
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
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
# Same model as v2/v3
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
# Data loading & client splits
# ────────────────────────────────────────────────────────────
def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return train, test


def split_for_clients(dataset, n_clients=10, samples_per_client=1000):
    """IID split: each client gets a random subset."""
    indices = np.random.permutation(len(dataset))
    clients = []
    for i in range(n_clients):
        idx = indices[i * samples_per_client:(i + 1) * samples_per_client]
        clients.append(Subset(dataset, idx))
    return clients


def flip_labels(subset, num_classes=10):
    """
    Adversarial label flip: replace every label y with (9 − y).
    This makes the client's gradients actively conflict with honest clients.
    """
    data, labels = [], []
    for x, y in subset:
        data.append(x)
        labels.append((num_classes - 1) - y)  # 0→9, 1→8, 2→7, ...
    data = torch.stack(data)
    labels = torch.tensor(labels)
    return TensorDataset(data, labels)


# ────────────────────────────────────────────────────────────
# Local training & FedAvg (same as v2/v3)
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


def federated_run(client_loaders, test_loader, rho, rounds=10, sample_seed=0):
    """
    Standard FedAvg: one global model. Each round, ρ fraction of clients
    (some honest, some adversarial) fork → train → average back.
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
    print("ADVERSARIAL FEDERATED EXPERIMENT (v4.0)")
    print("Testing P1-v4, P2-v4, P3-v4 from PRE_REGISTRATION_v4.md")
    print("Same model & FedAvg as v2/v3; 30% clients have flipped labels")
    print("=" * 64)

    train, test = load_mnist()
    test_loader = DataLoader(test, batch_size=256, shuffle=False)

    rhos = [0.1, 0.3, 0.5, 0.7, 1.0]
    n_seeds = 3
    n_adversarial = 3  # 30% of 10 clients

    # Build clients: 7 honest + 3 adversarial (label-flipped)
    client_subsets = split_for_clients(train, n_clients=10,
                                       samples_per_client=1000)

    print(f"\nClient assignments:")
    client_loaders = []
    for i, subset in enumerate(client_subsets):
        if i < n_adversarial:
            # Adversarial: flip labels
            flipped = flip_labels(subset)
            client_loaders.append(DataLoader(flipped, batch_size=32, shuffle=True))
            print(f"  Client {i}: ADVERSARIAL (labels flipped: y → 9−y)")
        else:
            client_loaders.append(DataLoader(subset, batch_size=32, shuffle=True))
            print(f"  Client {i}: honest")

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
        "experiment": "adversarial clients v4.0",
        "date": "2026-04-25",
        "setup": {
            "clients": 10,
            "adversarial_clients": n_adversarial,
            "adversarial_type": "label flip (y → 9−y)",
            "rounds": 10,
            "seeds": n_seeds,
            "samples_per_client": 1000,
            "model": "SmallCNN (same as v2/v3)",
            "normalization": "mean=0.1307, std=0.3081",
        },
        "results": results,
    }
    with open("../results/adversarial_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n✅ Results saved: ../results/adversarial_results.json")

    # ── Verdicts ────────────────────────────────────────────
    accs_all = [r["acc_mean"] for r in results]
    peak_idx = int(np.argmax(accs_all))
    peak_rho = results[peak_idx]["rho"]
    is_interior = 0 < peak_idx < len(results) - 1

    # P3-v4: is full-sync worse than partial?
    acc_full = results[-1]["acc_mean"]  # ρ=1.0
    acc_07   = results[3]["acc_mean"]   # ρ=0.7
    acc_05   = results[2]["acc_mean"]   # ρ=0.5
    p3_pass  = acc_full < max(acc_07, acc_05)

    verdicts = {
        "P1-v4": "PASS" if is_interior else "FAIL",
        "P2-v4": "PASS" if peak_rho < 1.0 else "FAIL",
        "P3-v4": "PASS" if p3_pass else "FAIL",
    }

    with open("../results/adversarial_verdicts.json", "w") as f:
        json.dump({
            "experiment": "adversarial clients v4.0",
            **verdicts,
            "peak_rho": peak_rho,
            "acc_at_peak": results[peak_idx]["acc_mean"],
            "acc_at_full_sync": acc_full,
        }, f, indent=2)

    print("\n" + "=" * 64)
    print("VERDICT (Pre-Registered)")
    print("=" * 64)
    print(f"  Peak ρ* = {peak_rho}  (acc = {results[peak_idx]['acc_mean']:.4f})")
    print(f"  Full sync acc (ρ=1.0) = {acc_full:.4f}")
    print()
    for pid, v in verdicts.items():
        sym = "✅" if v == "PASS" else "❌"
        print(f"  {sym} {pid}: {v}")
    print()

    if is_interior:
        print("  🔥 INTERIOR OPTIMUM FOUND — adversarial conflict creates it!")
        print(f"     Partial sync (ρ={peak_rho}) beats full sync by "
              f"{results[peak_idx]['acc_mean'] - acc_full:+.4f}")
    else:
        print("  Full sync still wins even with adversarial clients.")
        print("  The adversarial hypothesis is falsified.")
    print()

    # ── Plot ────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("v4 Adversarial: 30% Clients Have Flipped Labels",
                     fontsize=13, fontweight="bold")

        rs = [r["rho"] for r in results]
        accs = [r["acc_mean"] for r in results]
        stds = [r["acc_std"] for r in results]
        cs = [r["comm"] for r in results]

        axes[0].errorbar(rs, accs, yerr=stds, marker="o",
                         label="30% adversarial", capsize=3,
                         color="#E91E63", linewidth=2)
        axes[0].scatter([peak_rho], [accs[peak_idx]], s=120, zorder=5,
                        color="#E91E63", edgecolor="black", linewidth=1.5)
        axes[0].axvspan(0.1, 0.9, alpha=0.06, color="green")
        axes[0].set_xlabel("Sync density ρ")
        axes[0].set_ylabel("Test accuracy")
        axes[0].set_title("Accuracy vs. Sync Density")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(rs, cs, marker="s", color="#E91E63", linewidth=2)
        axes[1].set_xlabel("Sync density ρ")
        axes[1].set_ylabel("Communication cost (proxy)")
        axes[1].set_title("Communication Cost")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("../results/adversarial_results.png", dpi=120)
        print("✅ Plot saved: ../results/adversarial_results.png")
    except ImportError:
        print("⚠️  matplotlib not installed — skipping plot")


if __name__ == "__main__":
    main()
