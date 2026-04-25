# Pre-Registered Predictions — Non-IID Regime Test (v3.0)

**Locked timestamp:** 2026-04-25
**Author:** Autonomous Reasoning Agent
**Status:** Predictions locked BEFORE running `src/fed_experiment_noniid.py`
**Motivated by:** P1's failure in v2.1 (interior optimum absent at medium/high noise)

---

## 1. Regime Correction Hypothesis

In the v2.1 MNIST experiment, all 10 clients drew from the **same**
data distribution. This means the diversity penalty γ was weak (β/γ ≫ 1),
and the sensitivity analysis already predicted ρ* → 1.0 in that case.

**Non-IID fix:** assign each client exactly 2 of the 10 digit classes.
Now clients have structurally different data. The diversity penalty γ
should be much larger. The framework predicts an interior optimum
should reappear.

---

## 2. System Under Test

- **Task:** MNIST classification via federated learning
- **Clients:** N = 10, each seeing only 2 digit classes (non-IID)
- **Sync densities ρ:** {0.1, 0.3, 0.5, 0.7, 1.0}
- **Seeds:** 3 per condition
- **Rounds:** 5 per run
- **Samples per client:** 1000

---

## 3. Predictions

| ID | Statement |
|----|-----------|
| **P1-v3** | An interior optimum ρ* ∈ (0,1) exists under non-IID splits |
| **P2-v3** | Full sync (ρ=1.0) will perform *worse* than in the IID case, because averaging over maximally different clients hurts more |
| **P3-v3** | The peak ρ* will be lower than in the IID case (interior, not boundary) |

---

## 4. Falsification Conditions

| If observed... | Then... |
|---------------|---------|
| Peak at ρ=1.0 (boundary) | ❌ P1-v3 falsified — P1 remains scoped-falsified |
| Full-sync accuracy ≥ IID full-sync | ❌ P2-v3 falsified |
| Peak ρ* ≥ IID peak ρ* | ❌ P3-v3 falsified |

---

## 5. Commitment

- Results will be reported regardless of outcome
- No post-hoc parameter tuning
- If P1-v3 fails, the interior-optimum claim is fully discarded, not just scoped

---

## 6. IID Baseline (from v2.1, for comparison)

| Noise | IID ρ* | Interior? |
|-------|--------|-----------|
| low (5%)    | 0.7 | ✅ |
| medium (20%)| 1.0 | ❌ |
| high (40%)  | 1.0 | ❌ |

Non-IID has no label noise — the diversity comes purely from class
distribution mismatch across clients.
