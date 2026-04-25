# Pre-Registered Predictions — Adversarial Clients (v4.0)

**Locked timestamp:** 2026-04-25
**Author:** Autonomous Reasoning Agent
**Status:** Predictions locked BEFORE running `src/fed_experiment_adversarial.py`
**Motivated by:** v2 + v3 showed consensus dominates diversity in all standard regimes

---

## 1. Adversarial Hypothesis

v2 (IID + noise) and v3 (non-IID class splits) both showed ρ*=1.0 in
most conditions. The revised thesis states:

> "Optimal connectivity remains at full synchronization **unless agents
> produce conflicting or adversarial updates**."

v4 tests the "unless" clause directly. If 30% of clients have
**flipped labels** (every label replaced by 9−y), their gradients
actively conflict with honest clients. Averaging them in should
**harm** the global model — making full sync worse than partial sync.

---

## 2. System Under Test

- **Task:** MNIST classification via federated learning
- **Clients:** N = 10
  - 7 honest clients (IID, no noise)
  - 3 adversarial clients (all labels flipped: y → 9−y)
- **Sync densities ρ:** {0.1, 0.3, 0.5, 0.7, 1.0}
- **Seeds:** 3 per condition
- **Rounds:** 10
- **Samples per client:** 1000
- **Model:** SmallCNN (same as v2/v3)

---

## 3. Predictions

| ID | Statement |
|----|-----------|
| **P1-v4** | An interior optimum ρ* ∈ (0,1) exists — full sync is NOT optimal |
| **P2-v4** | ρ* < 1.0 — the peak shifts left because averaging adversarial clients hurts |
| **P3-v4** | Full-sync accuracy (ρ=1.0) is LOWER than partial-sync (ρ=0.5 or 0.7) |

---

## 4. Falsification Conditions

| If observed... | Then... |
|---------------|---------|
| Peak still at ρ=1.0 | ❌ P1-v4 and P2-v4 falsified — adversarial clients not harmful enough |
| ρ=1.0 accuracy ≥ ρ=0.7 accuracy | ❌ P3-v4 falsified |
| All predictions fail | The revised thesis is wrong — consensus dominates even under adversarial conflict |

---

## 5. Why This Is the Real Test

- v2 tested noise → consensus dominated
- v3 tested class heterogeneity → consensus dominated
- v4 tests **active conflict** → the one regime where diversity
  penalty γ should be large enough to overpower consensus benefit β

If the interior optimum doesn't appear here, it likely doesn't
appear in any practical federated learning scenario, and the
entire P1 line of inquiry should be permanently closed.

---

## 6. Commitment

- Results will be reported regardless of outcome
- No post-hoc tuning of the adversarial fraction (30% is locked)
- If all predictions fail, the adversarial hypothesis is discarded
