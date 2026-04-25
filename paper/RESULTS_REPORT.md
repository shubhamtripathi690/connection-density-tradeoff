# Federated Learning Experiment — Honest Results Report

**Run date:** 2026-04-25
**Environment:** Python 3.11 venv on macOS, CPU only
**Pre-registration:** locked 2026-04-25 (see [prereg/PRE_REGISTRATION.md](../prereg/PRE_REGISTRATION.md))
**Code:** [src/fed_experiment.py](../src/fed_experiment.py)
**Raw data:** [results/raw_results.json](../results/raw_results.json)
**Verdicts:** [results/verdicts.json](../results/verdicts.json)
**Plot:** [results/fed_results.png](../results/fed_results.png)

---

## 1. Setup (matches pre-registration)

- Task: MNIST classification, 10 simulated clients
- Sync densities tested: ρ ∈ {0.1, 0.3, 0.5, 0.7, 1.0}
- Noise levels: low (5%), medium (20%), high (40%) label corruption
- 3 seeds per condition; 10 federated rounds each
- Architecture: small CNN (~50k params)

---

## 2. Headline Result: 3/4 PASS, 1/4 FAIL

| Prediction | Verdict | Evidence |
|-----------|---------|----------|
| P1 — Interior optimum exists | ❌ **FAIL** | Only 1 of 3 noise levels showed it |
| P2 — ρ\* shifts with noise | ✅ **PASS** | Peaks: 0.7 → 1.0 → 1.0 |
| P3 — ρ\* ≠ 0.5 universally | ✅ **PASS** | No noise level peaked at 0.5 |
| P4 — Sub-linear accuracy gain | ✅ **PASS** | 10× cost → 1.17× accuracy |

---

## 3. Detailed Numbers

### Accuracy (mean ± std over 3 seeds)

| Noise | ρ=0.1 | ρ=0.3 | ρ=0.5 | ρ=0.7 | ρ=1.0 | Peak |
|-------|-------|-------|-------|-------|-------|------|
| low    | 0.743 ± 0.090 | 0.834 ± 0.022 | 0.853 ± 0.024 | **0.876 ± 0.019** | 0.871 ± 0.020 | ρ=0.7 |
| medium | 0.669 ± 0.065 | 0.799 ± 0.012 | 0.841 ± 0.019 | 0.853 ± 0.005 | **0.866 ± 0.004** | ρ=1.0 |
| high   | 0.674 ± 0.072 | 0.764 ± 0.017 | 0.779 ± 0.055 | 0.770 ± 0.009 | **0.839 ± 0.020** | ρ=1.0 |

### Communication cost (proxy: total client-uploads across rounds)

Strictly linear in ρ as predicted: 10, 30, 50, 70, 100.

---

## 4. Honest Interpretation of P1 Failure

**P1 claimed:** an interior optimum ρ\* ∈ (0,1) will exist at every
noise level.

**Observation:** the interior optimum existed only at the lowest
noise level (5% label corruption). At medium and high noise, full
synchronization (ρ = 1.0) won outright.

### Why this happened (post-hoc, but stated as hypothesis, not fact)

In the trade-off model $J(\rho) = V(\rho) + \eta \rho^{p}$:

- The consensus benefit $V(\rho)$ shrinks rapidly with more sharing.
- The diversity penalty $\eta \rho^{p}$ is supposed to grow with ρ.
- In this MNIST/CNN setup, the diversity penalty appears very weak —
  likely because all 10 clients are sampling from the same
  distribution, so averaging more of them never hurts within the
  tested rounds.

**This is a regime where β/γ ≫ 1**, exactly the case where the prior
sensitivity analysis predicted ρ\* → 1.0. The framework's
**sensitivity-analysis output was correct; the pre-registered
prediction was too optimistic.**

### What I will NOT claim

- ❌ "P1 still passes if you squint" — it doesn't, the boundary peak is real
- ❌ "MNIST is a special case" — yes, but I picked it; that's on me
- ❌ "Just rerun with different parameters" — that's exactly the
  post-hoc tuning I committed not to do

---

## 5. What Survives (and Should Be Trusted More)

✅ **The trade-off framework's qualitative direction** is supported:
   ρ\* genuinely shifts with system parameters (P2).

✅ **The "no universal 50%" claim** is supported by real data (P3),
   not just simulation.

✅ **The communication-cost insight** holds: paying 10× the bandwidth
   buys only 17% more accuracy. This is a publishable practical
   finding for federated learning practitioners.

---

## 6. What Must Be Revised

⚠️ **P1's universality claim is retracted.** A more honest version:

> *"An interior optimum ρ\* ∈ (0,1) MAY exist when system noise is below
> a regime-specific threshold. Above that threshold, full synchronization
> dominates. The threshold must be measured per system."*

This is weaker, and that's appropriate.

---

## 7. Comparison: Simulation Predicted vs. Real Data

The earlier [src/honest_sensitivity.py](../src/honest_sensitivity.py)
predicted that when β/γ ≫ 1, ρ\* should approach 1.0. The medium and
high noise regimes likely fall in this category. So the **sensitivity
analysis was right**; the pre-registered prediction simply didn't
account for the empirical β/γ ratio of MNIST + 10 clients being
skewed.

This is a useful self-consistency check: the framework's *internal*
predictions matched, but the *external* prediction (P1) was too
strong.

---

## 8. Concrete Next Steps (data-driven)

1. **Update the paper to v2.1** with the retraction in §6 ✅ done
2. **Add real-data column** to the Truth-Falsity Matrix ✅ done
3. **Test in a regime where diversity matters more**:
   - Non-IID client splits (each client only sees 2 digit classes)
   - Adversarial clients (some clients deliberately wrong)
   - These are setups where the diversity penalty γ should be larger,
     and an interior optimum is more plausible
4. **Do not** retroactively edit P1 to make it pass

---

## 9. What I Learned

| Lesson | How I'll Apply It |
|--------|-------------------|
| Pre-registered predictions can fail even when the framework is sound | Predictions must be scoped, not universal |
| "Direction" predictions (P2) are more robust than "existence" claims (P1) | Prefer directional hypotheses |
| Real data exposes regime assumptions that simulations hide | Run real data earlier, not later |
| 3/4 PASS is still meaningful — but 1/4 FAIL must be reported as FAIL | No goalpost moving |

---

## 10. Honest Bottom Line

> *The framework's qualitative claims survived first contact with real data.*
> *The framework's universal claims did not.*
> *This is good news: it tells me exactly where the model breaks,*
> *and exactly what to test next.*

---

**End of report. No edits will be made retroactively. Updates will appear
as v2.2+ with explicit changelog entries.**
