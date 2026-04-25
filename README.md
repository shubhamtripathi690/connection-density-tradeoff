# Connection-Density Trade-offs in Multi-Agent Systems

[![pre-registered](https://img.shields.io/badge/pre--registered-yes-brightgreen)]()
[![results](https://img.shields.io/badge/results-3%2F4%20PASS%2C%201%2F4%20FAIL-orange)]()
[![license](https://img.shields.io/badge/license-MIT-lightgrey)]()
[![demo](https://img.shields.io/badge/demo-notebook-blueviolet)](demo.ipynb)

> **How much should agents in a multi-agent system communicate?**
> I pre-registered four predictions, ran a real experiment, and reported
> the result honestly — including the one that failed.

---

## The result, up front

![Accuracy vs sync density across three noise levels](results/fed_results.png)

*Left: test accuracy vs. sync density ρ for three noise levels (3 seeds, error bars = std).
Right: communication cost is exactly linear while accuracy gain is sub-linear.*

---

## v3 update — Non-IID regime (2026-04-25)

![v2 IID vs v3 Non-IID comparison](results/v2_vs_v3_comparison.png)

**v3 verdict: 1/3 PASS, 2/3 FAIL — honestly reported.**

| ID | Prediction | Verdict | Evidence |
|----|-----------|---------|----------|
| **P1-v3** | Interior optimum in non-IID regime | ❌ **FAIL** | ρ\*=1.0 again — P1 fully discarded |
| **P2-v3** | Full-sync accuracy lower in non-IID | ✅ **PASS** | 0.701 vs 0.871 (IID low noise) — 20% drop |
| **P3-v3** | Peak ρ\* shifts left vs IID | ❌ **FAIL** | ρ\*=1.0 in both cases |

> **What this means:** Class-distribution mismatch alone isn't enough to create an
> interior optimum within 10 rounds of FedAvg. Full sync still wins — but it wins
> with much lower absolute accuracy (0.701 vs 0.871). The framework correctly
> predicted non-IID would be harder; it mis-predicted *where the optimum would land*.
> P1 is now fully discarded, not just scoped.

---

## v4 update — Adversarial clients (2026-04-25)

![v4 adversarial results](results/adversarial_results.png)

**v4 verdict: 0/3 PASS, 3/3 FAIL — honestly reported.**

| ID | Prediction | Verdict | Evidence |
|----|-----------|---------|----------|
| **P1-v4** | Interior optimum with adversarial clients | ❌ **FAIL** | ρ\*=1.0 still the peak (acc=0.791) |
| **P2-v4** | Peak ρ\* < 1.0 | ❌ **FAIL** | Peak is at boundary |
| **P3-v4** | Full sync accuracy < partial sync | ❌ **FAIL** | ρ=1.0 (0.791) > ρ=0.7 (0.737) |

> **What this means:** Even with 30% of clients sending actively conflicting
> gradients (all labels flipped), FedAvg's averaging still dilutes the poison
> enough that full sync wins. The interior-optimum hypothesis is **conclusively
> dead** across noise (v2), heterogeneity (v3), and adversarial conflict (v4).

| Experiment | Conditions | Interior optimum? |
|-----------|-----------|-------------------|
| v2 (IID + noise) | 3 noise levels × 5 ρ | Only at low noise (ρ\*=0.7) |
| v3 (non-IID class splits) | 10 clients × 5 ρ | ❌ No |
| v4 (30% adversarial) | 10 clients × 5 ρ | ❌ No |

> *"I showed that neither heterogeneity nor adversarial conflict is sufficient
> to penalize communication in standard FedAvg. Consensus is more robust than
> the theory predicted."*

---

## The story

### 1 — The hypothesis

Multi-agent systems face a fundamental tension: more communication
reduces noise through consensus, but erodes independent diversity.
The trade-off model predicts an **interior optimum** ρ\*:

```
J(ρ) = V(ρ)          +  η · ρᵖ
        ↑                  ↑
  consensus benefit    diversity cost
```

A prior simulation across 36 parameter regimes showed ρ\* ranges 0.10–1.00
and ρ\* = 0.5 occurs in only ~6% of cases — **not a universal law.**
See [`src/honest_sensitivity.py`](src/honest_sensitivity.py).

### 2 — The pre-registration

Four predictions were **locked before any real data was seen**:
([`prereg/PRE_REGISTRATION.md`](prereg/PRE_REGISTRATION.md))

| ID | Prediction |
|----|-----------|
| **P1** | Interior optimum ρ\* ∈ (0,1) exists at **every** noise level |
| **P2** | ρ\* shifts with client noise (more noise → higher ρ\*) |
| **P3** | ρ\* ≠ 0.5 universally |
| **P4** | Comm cost linear in ρ; accuracy gain sub-linear |

### 3 — The experiment

MNIST federated learning · 10 clients · 3 noise levels · 5 sync densities · 3 seeds
Code: [`src/fed_experiment.py`](src/fed_experiment.py)

### 4 — The results

| ID | Prediction | Verdict | What happened |
|----|-----------|---------|---------------|
| **P1** | Interior optimum at every noise level | ❌ **FAIL** | Low noise: ρ\*=0.7 ✓ · Medium & high: peaked at boundary ρ=1.0 |
| **P2** | ρ\* shifts with noise | ✅ **PASS** | Peaks moved 0.7 → 1.0 → 1.0 |
| **P3** | ρ\* ≠ 0.5 universally | ✅ **PASS** | No noise level peaked at 0.5 |
| **P4** | Comm linear, accuracy sub-linear | ✅ **PASS** | 10× bandwidth → only 1.17× accuracy |

Raw numbers (mean over 3 seeds):

| Noise | ρ=0.1 | ρ=0.3 | ρ=0.5 | ρ=0.7 | ρ=1.0 | **Peak** |
|-------|-------|-------|-------|-------|-------|----------|
| low (5%)    | 0.743 | 0.834 | 0.853 | **0.876** | 0.871 | **ρ=0.7** ← interior ✅ |
| medium (20%)| 0.669 | 0.799 | 0.841 | 0.853 | **0.866** | **ρ=1.0** ← boundary ❌ |
| high (40%)  | 0.674 | 0.764 | 0.779 | 0.770 | **0.839** | **ρ=1.0** ← boundary ❌ |

### 5 — The honest failure

P1 claimed an interior optimum at **every** noise level. It appeared
only at low noise. Why? In this setup all 10 clients draw from the same
distribution, so the diversity penalty γ is weak — the β/γ ratio is
large, and the sensitivity analysis already predicted ρ\* → 1.0 in that
case. **The framework's internal prediction was right; the external
pre-registered claim was too strong.**

**What I won't do:** retroactively redefine P1 to pass. The verdict in
[`results/verdicts.json`](results/verdicts.json) says `FAIL` and
[`paper/HONEST_PAPER.md`](paper/HONEST_PAPER.md) §6 is an explicit
retraction.

### 6 — The final thesis (v2 + v3 + v4)

> *"In standard FedAvg, consensus is remarkably robust. Full
> synchronization dominates across IID noise, non-IID class splits,
> and 30% adversarial label-flip clients. The interior-optimum
> hypothesis is dead for this architecture. The surviving insight:
> 10× communication cost buys only ~17% accuracy gain."*

Three experiments, twelve conditions, one answer: **full sync wins.**
The only exception was v2 low noise (ρ\*=0.7) — the mildest regime.
Even adversarial conflict (v4) wasn't enough to break it.

| Stage | What was tested | What we learned |
|-------|----------------|----------------|
| v1 | Simulation | Universal 50% rule → falsified |
| v2 | IID + noise | Interior optimum only at low noise; 3/4 PASS |
| v3 | Non-IID class splits | Consensus still dominates; 1/3 PASS |
| v4 | 30% adversarial clients | Consensus *still* dominates; 0/3 PASS |

> The theory didn't survive. But the process of killing it produced a
> sharper, more useful insight than the theory itself.

---

## Interactive demo

Explore the results without re-running the experiment:

```bash
pip install -r requirements.txt
jupyter notebook demo.ipynb
```

[`demo.ipynb`](demo.ipynb) loads [`results/raw_results.json`](results/raw_results.json),
re-plots every curve, and prints the verdict logic step-by-step —
runs in under 5 seconds, no GPU needed.

---

## Reproduce from scratch

```bash
git clone <this-repo>
cd connection-density-tradeoff

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd src
python fed_experiment.py        # ~5–10 min on CPU
python honest_sensitivity.py    # sensitivity sweep, ~1 min
```

See [`docs/REPRODUCING.md`](docs/REPRODUCING.md) for determinism notes.

---

## Repository layout

```
connection-density-tradeoff/
├── README.md                      ← you are here
├── demo.ipynb                     ← interactive results explorer ⭐
├── requirements.txt
├── CHANGELOG.md
│
├── prereg/
│   ├── PRE_REGISTRATION.md        ← v2.1 predictions (frozen — do not edit)
│   └── PRE_REGISTRATION_v3.md     ← v3.0 non-IID predictions (locked) ⬅ new
│
├── src/
│   ├── fed_experiment.py          ← IID experiment (v2.1)
│   ├── fed_experiment_noniid.py   ← non-IID regime test (v3.0)
│   ├── compare_v2_v3.py           ← generates 4-panel comparison plot
│   └── honest_sensitivity.py      ← simulation that falsified v1
│
├── results/
│   ├── fed_results.png            ← v2 IID accuracy + comm-cost plot
│   ├── raw_results.json           ← v2 IID raw numbers
│   ├── verdicts.json              ← v2 P1–P4 verdicts
│   ├── noniid_results.json        ← v3 non-IID raw numbers
│   ├── noniid_verdicts.json       ← v3 P1–P3 verdicts
│   └── v2_vs_v3_comparison.png   ← 4-panel comparison plot
│
├── paper/
│   ├── HONEST_PAPER.md            ← full paper v2.1 (with retraction §6)
│   ├── RESULTS_REPORT.md          ← detailed empirical writeup
│   └── TRUTH_FALSITY_MATRIX.md    ← what survived contact with data
│
└── docs/
    └── REPRODUCING.md
```

---

## Claims and non-claims

| ✅ This repo claims | ❌ This repo does not claim |
|--------------------|-----------------------------|
| In standard FedAvg, consensus dominates diversity — full sync wins | Interior optimum exists in any tested regime |
| This holds across IID noise, non-IID splits, AND 30% adversarial clients | A universal "50% rule" of any kind |
| 10× comm buys only 1.17× accuracy | Cross-domain unification |
| The interior-optimum hypothesis is dead for this architecture | That this generalizes beyond FedAvg+MNIST |
| Pre-registration prevents post-hoc spin | — |

---

## Methodology commitments

1. **Predictions locked first** — [`prereg/PRE_REGISTRATION.md`](prereg/PRE_REGISTRATION.md) timestamped before the run.
2. **No post-hoc tuning** — parameters fixed before the run; no sweep done after seeing results.
3. **Failure reported as failure** — `verdicts.json` → `"P1": "FAIL (at least one boundary peak)"`. Goalposts not moved.
4. **All artifacts public** — raw JSON, verdicts, plot, code, paper, pre-registration.

---

## What's next

**v2 (IID):** 3/4 PASS · **v3 (non-IID):** 1/3 PASS · **v4 (adversarial):** 0/3 PASS.

The interior-optimum line of inquiry (P1) is **closed.** Three experiments,
three regimes, same answer. What remains:

1. **Different architectures** — the result may be specific to SmallCNN+MNIST.
   Testing on CIFAR-10 or with a transformer backbone could reveal
   architecture-dependent trade-offs.
2. **Different aggregation** — FedAvg is robust to adversaries by construction
   (averaging dilutes poison). Byzantine-robust aggregation (Krum, trimmed
   mean) might show interior optima because they *amplify* rather than dilute
   diversity.
3. **Communication efficiency** — the P4 finding (10× cost → 1.17× accuracy)
   is the most practically useful result. A deeper study of the cost-accuracy
   Pareto frontier is the highest-leverage next step.

---

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite both the paper and the
retraction — they are inseparable.

---

## License

MIT — see [`LICENSE`](LICENSE).
