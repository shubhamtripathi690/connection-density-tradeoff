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

**Conclusion:** ρ\* = 0.5 is an artifact of symmetric coefficients.
Interior optima exist only when β/γ is moderate. **In high-β/γ
regimes, the boundary wins.**

Full numbers and analysis: [paper/RESULTS_REPORT.md](RESULTS_REPORT.md).

---

## 5. What Is Actually True

1. An intermediate optimum **can** exist in multi-agent systems with
   competing consensus and diversity pressures **— but only when β/γ
   is moderate** (the diversity penalty must be non-trivial).
2. The optimum's location is **system-specific**.
3. In multi-agent LLM ensembles, partial sharing **may** beat full
   sharing — but the regime must be measured first.

---

## 6. Retractions

### v2.1 RETRACTION

The pre-registered prediction P1 ("an interior optimum
ρ\* ∈ (0,1) exists for every noise level") is **partially falsified**
by the MNIST federated learning experiment. At medium and high
label-noise levels, full synchronization (ρ = 1.0) dominated.

**Revised claim:** "An interior optimum may exist when β/γ is
moderate. At high β/γ, the boundary wins. The threshold is
system-specific."

This retraction was made before any post-hoc parameter tuning, in
accordance with the pre-registration commitment.

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
- **Our own MNIST experiment shows full sync wins under noisy
  clients.**

A general theory must explain these. The present framework now
does so: high-β/γ regimes correctly predict full-sync dominance.

---

## 9. Conclusion

> *"In multi-agent systems with competing consensus and diversity
> pressures, an interior optimum in connection density may exist in
> moderate-β/γ regimes. In high-β/γ regimes, full synchronization
> dominates. The regime must be determined empirically per system."*

Less exciting than v2.0's "interior optimum exists." More accurate.

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
