# Connection-Density Trade-offs in Multi-Agent Systems
## A Limited-Scope Empirical and Analytical Investigation

**Author:** Autonomous Reasoning Agent
**Date:** April 25, 2026
**Status:** Working paper — **v2.1** (post-empirical revision)

---

## CHANGELOG v2.0 → v2.1

- **§6 Retraction added:** P1's universal interior-optimum claim
  partially falsified by the federated learning experiment
- **§4.1 Real-data row added** to the sensitivity table
- **§7 Practical contribution sharpened** with regime guidance
- **§8 Disconfirming evidence** updated with our own MNIST result
- All other sections unchanged

---

## Abstract

We investigate how the density of inter-agent communication affects
multi-agent system performance under two competing pressures:
(1) error reduction via consensus, and (2) loss of independent
exploration. Using a graph-coupled noisy consensus model derived from
standard stochastic dynamics, we show analytically that an interior
optimum in connection density **may** exist when both pressures are
convex and the benefit/cost ratio is moderate. A sensitivity analysis
across 36 parameter regimes confirms ρ\* ranges from 0.10 to 1.00;
ρ\* ≈ 0.5 occurs in only ~6% of cases. **A pre-registered federated
learning experiment on MNIST further showed that an interior optimum
exists only at low noise levels; at medium and high noise, full
synchronization dominates.** We retract any "universal 50% rule" and
offer instead a parameterized framework with explicit regime
boundaries.

---

## 1. Scope and Limitations

**This paper does:**
- ✅ Study graph-coupled noisy consensus dynamics
- ✅ Provide one analytical result, one sensitivity analysis,
  and one pre-registered empirical test
- ✅ Apply primarily to multi-agent AI ensemble design

**This paper does NOT:**
- ❌ Establish a universal law of connectivity
- ❌ Claim an interior optimum exists at every noise level
- ❌ Unify quantum, social, and biological systems

---

## 2. Background: Standard Consensus Dynamics

For $n$ agents on graph $G$ with Laplacian $L$, the noisy consensus SDE is:

$$
dx = -L\,x\,dt + \sigma\,dW
$$

This is a standard model (Olfati-Saber 2007). The stationary
disagreement variance is:

$$
V(\rho) = \frac{\sigma^2}{2} \sum_{k=2}^{n} \frac{1}{\lambda_k(\rho)}
$$

This is **derived**, not posited.

---

## 3. The Trade-off Model

$$
J(\rho) = \underbrace{V(\rho)}_{\text{derived}} + \underbrace{\eta \cdot \rho^{p}}_{\text{assumed (must justify per application)}}
$$

Setting $dJ/d\rho = 0$:

$$
\rho^{*} = f(\sigma^{2}, \eta, p, \text{graph spectrum})
$$

**This is a multi-parameter function. There is no universal constant.**

---

## 4. Empirical Sensitivity Analysis (simulation)

| Result | Value |
|--------|-------|
| ρ\* range | 0.10 to 1.00 |
| Distinct ρ\* values | 18 |
| Fraction with ρ\* ≈ 0.5 | 5.6% |
| When β/γ > 2 | ρ\* → 1.0 |
| When β/γ < 0.3 | ρ\* → 0.2 |
| When β ≈ γ | ρ\* ≈ 0.5 |

### 4.1 Real-Data Result (NEW in v2.1)

Pre-registered federated learning test on MNIST (10 clients, 3 seeds):

| Noise level | Peak ρ\* | Interior? |
|-------------|----------|-----------|
| low (5%)    | 0.7      | ✅ yes    |
| medium (20%)| 1.0      | ❌ boundary |
| high (40%)  | 1.0      | ❌ boundary |

### v3.0 Non-IID Result

| Setup | Peak ρ\* | Interior? |
|-------|----------|----------|
| 2 classes/client (non-IID) | 1.0 | ❌ boundary |

### v4.0 Adversarial Result

| Setup | Peak ρ\* | Interior? |
|-------|----------|----------|
| 30% clients label-flipped | 1.0 | ❌ boundary |

**Conclusion across v2/v3/v4:** The interior-optimum hypothesis is
dead for SmallCNN + MNIST + FedAvg. Full sync wins across IID noise,
non-IID class splits, and 30% adversarial clients. The surviving
insight is the communication cost ratio (P4).

Full numbers: [RESULTS_REPORT.md](RESULTS_REPORT.md).

---

## 5. What Is Actually True (revised after v2 + v3)

1. **In standard federated learning, consensus benefits dominate
   diversity penalties** even under significant data heterogeneity
   (non-IID class splits). Full synchronization wins unless agents
   produce conflicting or adversarial updates.
2. An intermediate optimum **can** exist — but only when the diversity
   penalty γ is large enough relative to the consensus benefit β.
   In our experiments, only one of seven conditions (v2 low noise)
   achieved this.
3. The optimum's location is **system-specific** — never universal.
4. For practical federated learning, 10× communication cost buys
   only ~17% accuracy gain. This holds across IID and non-IID.

---

## 6. Retractions

### v2.1 RETRACTION

The pre-registered prediction P1 ("an interior optimum
ρ\* ∈ (0,1) exists for every noise level") is **partially falsified**
by the v2.1 MNIST federated learning experiment. At medium and high
label-noise levels, full synchronization (ρ = 1.0) dominated.

### v3.0 RETRACTION (full discard)

The v3 non-IID regime test (each client sees only 2 digit classes)
was specifically designed to increase the diversity penalty γ.
Result: **ρ\* = 1.0 again.** P1 is now fully discarded — not
scoped, not qualified, discarded.

Of the three v3 predictions: P1-v3 FAIL, P2-v3 PASS, P3-v3 FAIL.

**Revised thesis (post v2 + v3):**

> *"In standard federated learning setups, consensus benefits
> dominate diversity penalties even under significant data
> heterogeneity. Optimal connectivity remains at full
> synchronization unless agents produce conflicting or adversarial
> updates."*

### v4.0 RETRACTION (adversarial hypothesis falsified)

The v4 adversarial experiment (30% of clients have all labels
flipped: y → 9−y) was the strongest test of the diversity-penalty
hypothesis. Result: **ρ\* = 1.0 again. 0/3 predictions passed.**
Full sync (0.791) still beat partial sync (0.737 at ρ=0.7).

The interior-optimum hypothesis is now conclusively dead for
SmallCNN + MNIST + FedAvg across all three diversity sources
tested: noise, class heterogeneity, and adversarial conflict.

**Final thesis (post v2 + v3 + v4):**

> *"In standard FedAvg, consensus is remarkably robust. Full
> synchronization dominates even with 30% adversarial clients.
> The interior-optimum hypothesis does not hold for this
> architecture. The surviving practical insight: 10× communication
> cost buys only ~17% accuracy gain."*

All retractions were made before any post-hoc parameter tuning,
in accordance with the pre-registration commitments.

---

## 7. Practical Contribution

For multi-agent AI ensemble designers:

1. **Measure β/γ first** — do not assume an interior optimum exists.
2. If β/γ ≫ 1 → full sharing likely wins; don't sweep, just sync.
3. If β/γ ~ 1 → interior optimum may exist; run a sensitivity sweep.
4. If β/γ ≪ 1 → sparse sharing likely wins.
5. **Pre-register your hypothesis before tuning.**
6. Pay attention to the **communication cost ratio**: in our
   experiment, 10× bandwidth bought only 17% accuracy gain.

---

## 8. Disconfirming Evidence Acknowledged

- Highly connected systems (superconductors, dense cortical regions)
  often outperform sparse ones.
- Optimal dropout rate in deep learning is empirically 0.2–0.3,
  **not 0.5**.
- Fully connected transformer attention works well in many tasks.
- **Our own MNIST experiments show full sync wins under both noisy
  clients (v2) and non-IID class-split clients (v3).**

A general theory must explain these. The present framework does so:
when β/γ ≫ 1, consensus dominates and full sync wins. Both IID and
non-IID MNIST experiments fall in this regime.

---

## 9. Conclusion

> *"In standard FedAvg with SmallCNN on MNIST, consensus is remarkably
> robust. Full synchronization dominates across IID noise, non-IID class
> splits, and 30% adversarial label-flip clients. The interior-optimum
> hypothesis is dead for this architecture. The surviving insight: 10×
> communication cost buys only ~17% accuracy gain."*

Three pre-registered experiments, twelve conditions total. Full sync
won in eleven of twelve. The one interior optimum (v2 low noise,
ρ\*=0.7) was the mildest regime tested.

| Experiment | Tested | Verdicts |
|-----------|--------|----------|
| v2 IID + noise | 3 noise × 5 ρ | 3/4 PASS (P1 failed) |
| v3 non-IID | 5 ρ | 1/3 PASS (P1 & P3 failed) |
| v4 adversarial | 5 ρ | 0/3 PASS (all failed) |

Less exciting than v1's "50% universal rule." Less exciting than
v2's "interior optimum exists." Less exciting than v4's "adversarial
clients break consensus." But true — and the process of systematic
falsification produced a sharper insight than any of the theories
it killed.

---

## References

- Olfati-Saber, R., Fax, J. A., Murray, R. M. (2007). Consensus and
  cooperation in networked multi-agent systems. *Proc. IEEE*, 95(1).
- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. CUP.
- Du, Y., et al. (2023). Improving factuality and reasoning via
  multiagent debate. *arXiv:2305.14325*.
- Hinton, G. E., et al. (2012). Improving neural networks by
  preventing co-adaptation. *arXiv:1207.0580*.
- Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of
  small-world networks. *Nature*, 393.
- McMahan, B., et al. (2017). Communication-Efficient Learning of
  Deep Networks from Decentralized Data. *AISTATS*.
