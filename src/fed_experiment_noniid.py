"""
Federated Learning Experiment — Non-IID Regime Test (v3.0)
===========================================================
Pre-registration: prereg/PRE_REGISTRATION_v3.md
Hypothesis: In a non-IID setup (each client sees only 2 digit classes),
the diversity penalty γ is much larger, so an interior optimum ρ* ∈ (0,1)
should re-emerge — rehabilitating P1 under a scoped form.

Run: python fed_experiment_noniid.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
import time

DEVICE = "cpu"


# ────────────────────────────────────────────────────────────
# Model
# ────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(26 * 26 * 16, 10),
        )

    def forward(self, x):
        return self.net(x)


# ────────────────────────────────────────────────────────────
# Non-IID Split — each client gets exactly 2 digit classes
# ────────────────────────────────────────────────────────────
def create_noniid_splits(dataset, num_clients=10):
    """
    Assign 2 digits per client so clients have maximally different
    data distributions — this is where the diversity penalty γ is large.
    """
    targets = np.array(dataset.targets)
    digit_groups = [
        [0, 1], [2, 3], [4, 5], [6, 7], [8, 9],
        [0, 2], [1, 3], [4, 6], [5, 7], [8, 0],
    ]
    client_indices = []
    for group in digit_groups[:num_clients]:
        idx = np.where(np.isin(targets, group))[0]
        client_indices.append(idx)
    return client_indices


# ────────────────────────────────────────────────────────────
# Local training
# ────────────────────────────────────────────────────────────
def train_local(model, loader, epochs=1):
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()


# ────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────
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
# Federated averaging (over selected subset of clients)
# ────────────────────────────────────────────────────────────
def average_models(models):
    avg = SimpleCNN()
    sd = avg.state_dict()
    for key in sd:
        sd[key] = torch.stack(
            [m.state_dict()[key].float() for m in models]
        ).mean(dim=0)
    avg.load_state_dict(sd)
    return avg


# ────────────────────────────────────────────────────────────
# Single experiment run
# ────────────────────────────────────────────────────────────
def run_experiment(rho, seed=42, rounds=5, samples_per_client=1000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    transform = transforms.ToTensor()
    train_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=256)

    client_indices = create_noniid_splits(train_data)
    clients = []
    for idx in client_indices:
        subset = Subset(train_data, idx[:samples_per_client])
        clients.append(DataLoader(subset, batch_size=32, shuffle=True))

    models = [SimpleCNN() for _ in range(len(clients))]
    start = time.time()

    for _ in range(rounds):
        # Local training on each client's biased data
        for i, loader in enumerate(clients):
            train_local(models[i], loader)

        # Partial sync: only ρ fraction of clients contribute to global model
        num_sync = max(1, int(rho * len(models)))
        selected = random.sample(models, num_sync)
        global_model = average_models(selected)

        # Broadcast back to all clients
        for i in range(len(models)):
            models[i].load_state_dict(global_model.state_dict())

    acc = evaluate(models[0], test_loader)
    return acc, time.time() - start


# ────────────────────────────────────────────────────────────
# Sweep across ρ values
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rhos = [0.1, 0.3, 0.5, 0.7, 1.0]
    n_seeds = 3

    print("=" * 56)
    print("NON-IID FEDERATED EXPERIMENT (v3.0)")
    print("Pre-registration: prereg/PRE_REGISTRATION_v3.md")
    print("Hypothesis: non-IID splits → large γ → interior ρ*")
    print("=" * 56)

    results = {}
    for rho in rhos:
        accs = []
        for seed in range(n_seeds):
            acc, t = run_experiment(rho, seed=seed)
            accs.append(acc)
            print(f"  ρ={rho:.1f} seed={seed} | acc={acc:.4f} | time={t:.1f}s")
        results[rho] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}

    print()
    print("─" * 56)
    print(f"  {'ρ':>4}  {'mean acc':>10}  {'std':>8}  {'peak?':>6}")
    print("─" * 56)
    peak_rho = max(results, key=lambda r: results[r]["mean"])
    for rho, v in results.items():
        marker = "★ PEAK" if rho == peak_rho else ""
        print(f"  {rho:>4.1f}  {v['mean']:>10.4f}  {v['std']:>8.4f}  {marker}")
    print("─" * 56)

    is_interior = 0 < list(results.keys()).index(peak_rho) < len(rhos) - 1
    verdict = "✅ INTERIOR OPTIMUM — P1 rehabilitated (scoped)" \
              if is_interior else \
              "❌ BOUNDARY PEAK — P1 remains falsified"
    print(f"\n  ρ* = {peak_rho}  →  {verdict}\n")
