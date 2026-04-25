"""
Federated Learning — Adversarial Fraction Sweep (v5.0)
=======================================================
Pre-registration: prereg/PRE_REGISTRATION_v5.md

Maps the (q, ρ) phase diagram: at what adversarial fraction q
does FedAvg consensus break?

7 q values × 5 ρ values × 3 seeds = 105 runs
Same SmallCNN + FedAvg as v2/v3/v4.

Run: python fed_experiment_q_sweep.py
Time: ~30–40 minutes on CPU
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ────────────────────────────────────────────────────────────
# Same model as v2/v3/v4
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
# Data
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
    indices = np.random.permutation(len(dataset))
    clients = []
    for i in range(n_clients):
        idx = indices[i * samples_per_client:(i + 1) * samples_per_client]
        clients.append(Subset(dataset, idx))
    return clients


def flip_labels_dataset(subset, num_classes=10):
    """Adversarial: flip every label y → (9 − y)."""
    data, labels = [], []
    for x, y in subset:
        data.append(x)
        labels.append((num_classes - 1) - y)
    return TensorDataset(torch.stack(data), torch.tensor(labels))


# ────────────────────────────────────────────────────────────
# FedAvg primitives (same as v2/v3/v4)
# ────────────────────────────────────────────────────────────
def local_train(model, loader, epochs=1, lr=0.01):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
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
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def federated_run(client_loaders, test_loader, rho, rounds=10, sample_seed=0):
    np.random.seed(sample_seed)
    global_model = SmallCNN().to(DEVICE)
    n_clients = len(client_loaders)
    n_participants = max(1, int(round(rho * n_clients)))

    for _ in range(rounds):
        chosen = np.random.choice(n_clients, n_participants, replace=False)
        local_states = []
        for c in chosen:
            local = SmallCNN().to(DEVICE)
            local.load_state_dict(global_model.state_dict())
            state = local_train(local, client_loaders[c], epochs=1)
            local_states.append(state)
        global_model.load_state_dict(average_states(local_states))

    return evaluate(global_model, test_loader)


# ────────────────────────────────────────────────────────────
# Main: q × ρ sweep
# ────────────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("ADVERSARIAL FRACTION SWEEP (v5.0)")
    print("Mapping the (q, ρ) phase diagram of FedAvg robustness")
    print("Pre-registration: prereg/PRE_REGISTRATION_v5.md")
    print("=" * 64)

    train, test = load_mnist()
    test_loader = DataLoader(test, batch_size=256, shuffle=False)

    q_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    rhos = [0.1, 0.3, 0.5, 0.7, 1.0]
    n_seeds = 3
    n_clients = 10
    total_runs = len(q_values) * len(rhos) * n_seeds

    # Build IID client splits ONCE
    np.random.seed(42)
    client_subsets = split_for_clients(train, n_clients=n_clients,
                                       samples_per_client=1000)

    results = []
    run_count = 0

    for q in q_values:
        n_adv = int(q * n_clients)
        print(f"\n{'─' * 64}")
        print(f"  q = {q:.1f}  ({n_adv} adversarial / {n_clients - n_adv} honest)")
        print(f"{'─' * 64}")

        # Build loaders: first n_adv are adversarial
        client_loaders = []
        for i, subset in enumerate(client_subsets):
            if i < n_adv:
                flipped = flip_labels_dataset(subset)
                client_loaders.append(DataLoader(flipped, batch_size=32, shuffle=True))
            else:
                client_loaders.append(DataLoader(subset, batch_size=32, shuffle=True))

        for rho in rhos:
            t0 = time.time()
            accs = []
            for s in range(n_seeds):
                torch.manual_seed(s)
                acc = federated_run(client_loaders, test_loader,
                                    rho, rounds=10, sample_seed=s)
                accs.append(acc)
            acc_mean = float(np.mean(accs))
            acc_std = float(np.std(accs))
            run_count += n_seeds
            elapsed = time.time() - t0
            print(f"    ρ={rho:.1f}  acc={acc_mean:.4f}±{acc_std:.4f}  "
                  f"[{run_count}/{total_runs}]  {elapsed:.1f}s")
            results.append({
                "q": q,
                "rho": rho,
                "acc_mean": acc_mean,
                "acc_std": acc_std,
            })

    # ── Save ────────────────────────────────────────────────
    os.makedirs("../results", exist_ok=True)
    out = {
        "experiment": "adversarial fraction sweep v5.0",
        "date": "2026-04-26",
        "setup": {
            "clients": n_clients,
            "q_values": q_values,
            "rhos": rhos,
            "rounds": 10,
            "seeds": n_seeds,
            "adversarial_type": "label flip (y → 9−y)",
            "model": "SmallCNN (same as v2/v3/v4)",
        },
        "results": results,
    }
    with open("../results/q_sweep_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Results saved: ../results/q_sweep_results.json")

    # ── Verdicts ────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("ANALYSIS")
    print("=" * 64)

    # Build lookups: (q, rho) → acc_mean / acc_std
    lookup     = {(r["q"], r["rho"]): r["acc_mean"] for r in results}
    lookup_std = {(r["q"], r["rho"]): r["acc_std"]  for r in results}

    # ── Find peak ρ* per q (with significance check) ──────
    # A non-boundary peak only counts if it beats ρ=1.0 by more
    # than 1 pooled standard deviation — otherwise it's noise.
    SIGNIFICANCE_MARGIN = 1.0  # number of pooled-stds required

    print(f"\n  {'q':>4}  {'ρ*':>5}  {'acc@ρ*':>8}  {'acc@1.0':>8}"
          f"  {'gap':>7}  {'threshold':>9}  {'interior?':>10}")
    print("  " + "─" * 68)
    peak_rhos = {}
    for q in q_values:
        accs_at_q = {rho: lookup[(q, rho)] for rho in rhos}
        raw_rho_star = max(accs_at_q, key=accs_at_q.get)
        raw_idx = rhos.index(raw_rho_star)

        # Check significance: does the raw peak beat ρ=1.0 by enough?
        acc_peak = accs_at_q[raw_rho_star]
        acc_full = accs_at_q[1.0]
        std_peak = lookup_std[(q, raw_rho_star)]
        std_full = lookup_std[(q, 1.0)]
        pooled_std = np.sqrt((std_peak**2 + std_full**2) / 2)
        gap = acc_peak - acc_full
        threshold = SIGNIFICANCE_MARGIN * pooled_std

        if 0 < raw_idx < len(rhos) - 1 and gap > threshold:
            # Genuine interior optimum — gap exceeds noise
            rho_star = raw_rho_star
            sig_label = "✅ YES (sig)"
        else:
            # Either boundary, or peak is within noise of ρ=1.0
            rho_star = 1.0
            sig_label = "" if raw_rho_star == 1.0 else "(noise)"

        peak_rhos[q] = rho_star
        print(f"  {q:>4.1f}  {rho_star:>5.1f}  {acc_peak:>8.4f}  {acc_full:>8.4f}"
              f"  {gap:>+7.4f}  {threshold:>9.4f}  {sig_label}")

    # ── P1-v5: does a critical q_c exist? ──────────────────
    q_c = None
    for q in q_values:
        if peak_rhos[q] < 1.0:
            q_c = q
            break

    print()
    if q_c is not None:
        print(f"  🔥 P1-v5 PASS: critical fraction q_c = {q_c}")
        print(f"     At q≥{q_c}, full sync is no longer optimal!")
    else:
        print(f"  ❌ P1-v5 FAIL: full sync wins at every q value")

    # ── P2-v5: at q≥0.5, does ρ=1.0 lose to best partial? ─
    if 0.5 in [r["q"] for r in results]:
        acc_05_full = lookup[(0.5, 1.0)]
        best_partial_05 = max(lookup[(0.5, rho)] for rho in rhos if rho < 1.0)
        p2 = acc_05_full < best_partial_05
        print(f"  {'✅' if p2 else '❌'} P2-v5: at q=0.5, "
              f"ρ=1.0 ({acc_05_full:.4f}) {'<' if p2 else '≥'} "
              f"best partial ({best_partial_05:.4f})")

    # ── P3-v5: at q≥0.7, does accuracy collapse (<30%)? ────
    if 0.7 in [r["q"] for r in results]:
        best_at_07 = max(lookup[(0.7, rho)] for rho in rhos)
        p3 = best_at_07 < 0.30
        print(f"  {'✅' if p3 else '❌'} P3-v5: at q=0.7, "
              f"best acc = {best_at_07:.4f} "
              f"({'<' if p3 else '≥'} 0.30 threshold)")

    # ── P4-v5: does ρ*=1.0 hold for q<0.3? ─────────────────
    p4 = all(peak_rhos[q] == 1.0 for q in q_values if q < 0.3)
    print(f"  {'✅' if p4 else '❌'} P4-v5: ρ*=1.0 for all q<0.3? {p4}")

    # ── Save verdicts ───────────────────────────────────────
    verdicts = {
        "experiment": "adversarial fraction sweep v5.0 (corrected)",
        "significance_rule": "interior peak must beat rho=1.0 by >1 pooled std",
        "q_c": q_c,
        "peak_rhos": {str(q): peak_rhos[q] for q in q_values},
        "P1-v5": "PASS" if q_c is not None else "FAIL",
        "P2-v5": "PASS" if (0.5 in [r["q"] for r in results] and
                             lookup[(0.5, 1.0)] < max(lookup[(0.5, rho)] for rho in rhos if rho < 1.0))
                 else "FAIL",
        "P3-v5": "PASS" if (0.7 in [r["q"] for r in results] and
                             max(lookup[(0.7, rho)] for rho in rhos) < 0.30)
                 else "FAIL",
        "P4-v5": "PASS" if p4 else "FAIL",
    }
    with open("../results/q_sweep_verdicts.json", "w") as f:
        json.dump(verdicts, f, indent=2)
    print(f"\n✅ Verdicts saved: ../results/q_sweep_verdicts.json")


if __name__ == "__main__":
    main()
