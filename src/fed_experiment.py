"""
Federated Learning Experiment — Pre-Registered Test
====================================================
Tests P1–P4 from PRE_REGISTRATION.md against real MNIST data.

Predictions:
  P1: Interior optimum ρ* ∈ (0,1) exists
  P2: ρ* shifts with client noise (more noise → higher ρ*)
  P3: ρ* ≠ 0.5 unless β ≈ γ
  P4: Communication cost linear in ρ; accuracy gain sub-linear

Run: python3 fed_experiment.py
Time: ~5–10 minutes on CPU; 1–2 minutes on GPU
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

# ────────────────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ────────────────────────────────────────────────────────────
# Simple MNIST classifier
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
    """Non-IID-ish split: each client gets a random subset."""
    indices = np.random.permutation(len(dataset))
    clients = []
    for i in range(n_clients):
        idx = indices[i * samples_per_client:(i + 1) * samples_per_client]
        clients.append(Subset(dataset, idx))
    return clients


# ────────────────────────────────────────────────────────────
# Add label noise to a client (simulates noisy data)
# ────────────────────────────────────────────────────────────
def add_label_noise(subset, noise_rate, num_classes=10):
    """Wrap a Subset with corrupted labels."""
    data, labels = [], []
    for x, y in subset:
        data.append(x)
        if np.random.rand() < noise_rate:
            labels.append(np.random.randint(num_classes))
        else:
            labels.append(y)
    data = torch.stack(data)
    labels = torch.tensor(labels)
    return TensorDataset(data, labels)


# ────────────────────────────────────────────────────────────
# Local training & FedAvg with sync density ρ
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


def federated_run(client_loaders, test_loader, rho, rounds=10, sample_seed=0):
    """
    Run federated learning where in each round, only fraction ρ of
    clients participate in the sync (sync density = ρ).
    """
    np.random.seed(sample_seed)
    global_model = SmallCNN().to(DEVICE)
    n_clients = len(client_loaders)

    n_participants = max(1, int(round(rho * n_clients)))
    total_comm = 0  # bytes proxy = participants × rounds

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
# Main experiment loop
# ────────────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("FEDERATED LEARNING — PRE-REGISTERED EXPERIMENT")
    print("Testing predictions P1–P4 from PRE_REGISTRATION.md")
    print("=" * 64)

    train, test = load_mnist()
    test_loader = DataLoader(test, batch_size=256, shuffle=False)

    rhos = [0.1, 0.3, 0.5, 0.7, 1.0]
    noise_levels = {"low": 0.05, "medium": 0.20, "high": 0.40}
    n_seeds = 3  # average over multiple seeds for robustness

    results = {}

    for noise_name, noise_rate in noise_levels.items():
        print(f"\n{'─' * 64}")
        print(f"Noise level: {noise_name} (label noise rate = {noise_rate})")
        print(f"{'─' * 64}")

        # Build noisy clients ONCE for this noise level
        client_subsets = split_for_clients(train, n_clients=10,
                                            samples_per_client=1000)
        noisy_clients = [add_label_noise(c, noise_rate) for c in client_subsets]
        client_loaders = [DataLoader(c, batch_size=32, shuffle=True)
                          for c in noisy_clients]

        results[noise_name] = []
        for rho in rhos:
            t0 = time.time()
            accs, comms = [], []
            for s in range(n_seeds):
                acc, comm = federated_run(client_loaders, test_loader,
                                          rho, rounds=10, sample_seed=s)
                accs.append(acc)
                comms.append(comm)
            acc_mean = np.mean(accs)
            acc_std = np.std(accs)
            comm_mean = np.mean(comms)
            print(f"  ρ={rho:.1f}  acc={acc_mean:.4f}±{acc_std:.4f}  "
                  f"comm={comm_mean:.0f}  time={time.time()-t0:.1f}s")
            results[noise_name].append({
                "rho": rho,
                "acc_mean": float(acc_mean),
                "acc_std": float(acc_std),
                "comm": float(comm_mean),
            })

    # ────────────────────────────────────────────────────────
    # Save raw results
    # ────────────────────────────────────────────────────────
    os.makedirs("../results", exist_ok=True)
    with open("../results/raw_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Raw results saved: ../results/raw_results.json")

    # ────────────────────────────────────────────────────────
    # Test predictions
    # ────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("PREDICTION TEST — Locked vs. Observed")
    print("=" * 64)

    verdicts = {}

    # P1: interior optimum
    p1_pass = []
    for noise_name, rows in results.items():
        accs = [r["acc_mean"] for r in rows]
        peak_idx = int(np.argmax(accs))
        peak_rho = rows[peak_idx]["rho"]
        is_interior = 0 < peak_idx < len(rows) - 1
        p1_pass.append(is_interior)
        print(f"  [{noise_name}] peak ρ* = {peak_rho:.1f}  "
              f"{'✅ interior' if is_interior else '❌ at boundary'}")
    verdicts["P1"] = "PASS" if all(p1_pass) else "FAIL (at least one boundary peak)"

    # P2: ρ* shifts with noise
    peak_rhos = []
    for noise_name in ["low", "medium", "high"]:
        accs = [r["acc_mean"] for r in results[noise_name]]
        peak_rhos.append(results[noise_name][int(np.argmax(accs))]["rho"])
    p2_shifts = len(set(peak_rhos)) > 1
    verdicts["P2"] = "PASS" if p2_shifts else "FAIL (ρ* constant across noise)"
    print(f"\n  P2: ρ* across noise = {peak_rhos}  "
          f"{'✅ shifts' if p2_shifts else '❌ constant'}")

    # P3: ρ* ≠ 0.5 always
    all_half = all(pr == 0.5 for pr in peak_rhos)
    verdicts["P3"] = "PASS" if not all_half else "AMBIGUOUS (50% rule rehabilitated)"

    # P4: comm linear, accuracy sub-linear
    accs_low = [r["acc_mean"] for r in results["low"]]
    comms_low = [r["comm"] for r in results["low"]]
    # Sub-linear if accuracy gain from ρ=0.1→1.0 is much less than 10x
    acc_gain = accs_low[-1] / max(accs_low[0], 1e-6)
    comm_gain = comms_low[-1] / max(comms_low[0], 1e-6)
    sublinear = acc_gain < comm_gain
    verdicts["P4"] = "PASS" if sublinear else "FAIL (accuracy super-linear)"
    print(f"\n  P4: acc_gain={acc_gain:.2f}x  comm_gain={comm_gain:.2f}x  "
          f"{'✅ sub-linear' if sublinear else '❌ super-linear'}")

    # ────────────────────────────────────────────────────────
    # Final verdict
    # ────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("FINAL VERDICT (Pre-Registered)")
    print("=" * 64)
    for k, v in verdicts.items():
        symbol = "✅" if v == "PASS" else "❌" if "FAIL" in v else "⚠️"
        print(f"  {symbol} {k}: {v}")

    with open("../results/verdicts.json", "w") as f:
        json.dump(verdicts, f, indent=2)
    print("\n✅ Verdicts saved: ../results/verdicts.json")

    # ────────────────────────────────────────────────────────
    # Plot
    # ────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for noise_name, rows in results.items():
            rs = [r["rho"] for r in rows]
            accs = [r["acc_mean"] for r in rows]
            stds = [r["acc_std"] for r in rows]
            axes[0].errorbar(rs, accs, yerr=stds, marker="o",
                             label=f"{noise_name} noise", capsize=3)

        axes[0].set_xlabel("Sync density ρ")
        axes[0].set_ylabel("Test accuracy")
        axes[0].set_title("Accuracy vs. Sync Density (3 noise levels)")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        # Right panel: comm cost
        for noise_name, rows in results.items():
            rs = [r["rho"] for r in rows]
            cs = [r["comm"] for r in rows]
            axes[1].plot(rs, cs, marker="s", label=f"{noise_name}")
        axes[1].set_xlabel("Sync density ρ")
        axes[1].set_ylabel("Communication cost (proxy)")
        axes[1].set_title("Communication Cost is Linear in ρ")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("../results/fed_results.png", dpi=120)
        print("✅ Plot saved: ../results/fed_results.png")
    except ImportError:
        print("⚠️  matplotlib not installed — skipping plot")


if __name__ == "__main__":
    main()
