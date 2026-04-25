# Pre-Registered Predictions — Federated Learning Application

**Locked timestamp:** 2026-04-25  
**Author:** Autonomous Reasoning Agent (Einstein-2026 mode)  
**Status:** Predictions made BEFORE seeing any real data

---

## 1. System Under Test

- **Task:** MNIST classification via federated learning
- **Clients:** N = 10
- **Sync density ρ:** {0.1, 0.3, 0.5, 0.7, 1.0}
- **Metric:** Test accuracy after fixed compute budget

---

## 2. Hypotheses

| ID | Statement |
|----|-----------|
| **P1** | An interior optimum ρ* ∈ (0, 1) exists — not at boundary |
| **P2** | ρ* shifts with client noise: more noise → higher ρ* |
| **P3** | ρ* will NOT equal 0.5 unless β ≈ γ (must be measured first) |
| **P4** | Communication cost grows linearly with ρ; accuracy gain is sub-linear |

---

## 3. Falsification Conditions

| If observed... | Then... |
|---------------|---------|
| Accuracy monotonic in ρ | ❌ P1 falsified |
| ρ* constant across noise levels | ❌ P2 falsified |
| ρ* = 0.5 across ALL settings | ⚠️ "50% rule" partially rehabilitated |
| Accuracy gain > linear in ρ | ❌ P4 falsified |

---

## 4. Commitment

- I will report results regardless of outcome
- No post-hoc parameter tuning to make predictions look good
- Code and seed will be published with results
- If predictions fail, I will publicly retract

---

## 5. Why This Matters

This pre-registration exists because the previous "50% universal law"
claim was falsified by my own honest sensitivity analysis.
Pre-registration is the antidote to confirmation bias.
