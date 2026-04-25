# Pre-Registered Predictions — Adversarial Fraction Sweep (v5.0)

**Locked timestamp:** 2026-04-26
**Author:** Autonomous Reasoning Agent
**Status:** Predictions locked BEFORE running `src/fed_experiment_q_sweep.py`
**Motivated by:** v2–v4 showed full sync wins even at 30% adversarial. Where does it break?

---

## 1. Research Question

v4 showed FedAvg survives 30% adversarial clients. But at what
adversarial fraction q does consensus actually break? This experiment
maps the **robustness boundary** — the (q, ρ) phase diagram.

---

## 2. System Under Test

- **Task:** MNIST classification via federated learning
- **Clients:** N = 10
- **Adversarial fraction q:** {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9}
  - Adversarial clients have all labels flipped: y → 9−y
- **Sync densities ρ:** {0.1, 0.3, 0.5, 0.7, 1.0}
- **Seeds:** 3 per condition
- **Rounds:** 10
- **Samples per client:** 1000
- **Model:** SmallCNN (same as v2/v3/v4)
- **Total runs:** 7 × 5 × 3 = 105

---

## 3. Predictions

| ID | Statement |
|----|-----------|
| **P1-v5** | There exists a critical fraction q_c where full sync (ρ=1.0) stops being optimal |
| **P2-v5** | At q ≥ 0.5 (majority adversarial), accuracy at ρ=1.0 drops below ρ=0.5 |
| **P3-v5** | At q ≥ 0.7, the system collapses — accuracy approaches random (≈10%) at all ρ |
| **P4-v5** | For q < q_c, the ρ*=1.0 result from v2/v3/v4 is reproduced |

---

## 4. Falsification Conditions

| If observed... | Then... |
|---------------|---------|
| ρ=1.0 remains optimal at ALL q values (even q=0.9) | ❌ P1-v5 falsified — FedAvg is indestructible in this setup |
| ρ=1.0 accuracy ≥ ρ=0.5 accuracy at q=0.5 | ❌ P2-v5 falsified |
| Accuracy stays well above random (>30%) at q=0.7 | ❌ P3-v5 falsified |
| ρ*≠1.0 appears at any q < 0.3 | ❌ P4-v5 falsified (contradicts v2/v3/v4) |

---

## 5. What This Maps

The output is a **phase diagram** with:
- x-axis: sync density ρ (communication)
- y-axis: adversarial fraction q (corruption)
- color: test accuracy

This shows three predicted regions:
- 🟢 **Low corruption (q ≤ 0.3):** FedAvg robust, ρ*=1.0
- 🟡 **Mid corruption (q ≈ 0.4–0.6):** performance degrades, interior optimum may appear
- 🔴 **High corruption (q ≥ 0.7):** FedAvg collapses, full sync becomes harmful

---

## 6. Commitment

- Results will be reported regardless of outcome
- No post-hoc tuning of q values or ρ grid
- If all predictions fail, the robustness-boundary hypothesis is discarded
