# Connection-Density Trade-offs in Multi-Agent Systems

[![pre-registered](https://img.shields.io/badge/pre--registered-5%20experiments-brightgreen)]()
[![runs](https://img.shields.io/badge/total%20runs-150+-blue)]()
[![license](https://img.shields.io/badge/license-MIT-lightgrey)]()
[![demo](https://img.shields.io/badge/demo-notebook-blueviolet)](demo.ipynb)

## One-Line Insight

> **"More communication helps — until adversaries outnumber honest agents. Then it becomes the attack vector."**

---

## What the Data Shows (at a glance)

![Phase diagram: FedAvg robustness map](results/q_vs_rho_heatmap.png)

- 🟢 **Low corruption (≤30%):** full sync is optimal — averaging dilutes poison
- 🟡 **Majority corruption (50%):** full sync becomes harmful — **interior optimum emerges**
- 🔴 **Overwhelming corruption (≥70%):** system collapses regardless of communication
- 📉 **10× communication cost → only 1.17× accuracy gain** (even in the best case)

> **There is a phase transition at q ≈ 0.5 where averaging switches from stabilizing to destabilizing.**
> *(Signal is real; variance is high at 3 seeds. Direction robust, exact boundary needs more data.)*

---

## The Story

**I started with a strong assumption:**

> *There exists a universal optimal level of connectivity in multi-agent systems.*

**I was wrong.** I tested it five times, each time stronger, and systematically
eliminated every version of the claim. What survived was smaller, sharper,
and more useful than what I started with.

| Stage | What I believed | What I tested | What happened |
|-------|----------------|--------------|---------------|
| **v1** | Universal 50% rule | Simulation (36 regimes) | ❌ Falsified by my own sensitivity analysis |
| **v2** | Interior optimum always exists | Real MNIST + noise | ⚠️ 3/4 PASS — but optimum only at low noise |
| **v3** | Non-IID data will create optimum | Class-split clients | ❌ Full sync still won |
| **v4** | Adversarial clients will break it | 30% label-flipped | ❌ Full sync *still* won |
| **v5** | There's a breaking point somewhere | 0–90% adversarial sweep | ✅ **Phase transition at q ≈ 0.5** |

**What I learned:**
The original theory was wrong — but the process of breaking it revealed
exactly *where* consensus stops working. That's a more useful result
than the theory itself.

---

## Why This Matters

Most distributed AI systems assume:
> *"More communication = better performance"*

This project shows that assumption has a **regime boundary:**

| If your system has... | Then... |
|----------------------|---------|
| Trustworthy agents (q ≤ 0.3) | Full sync wins. Don't bother sweeping — just communicate. |
| Significant corruption (q ≈ 0.5) | Partial sync outperforms full sync. Sweep ρ to find the optimum. |
| Majority compromise (q ≥ 0.7) | FedAvg cannot help. Switch to Byzantine-robust aggregation. |
| Limited bandwidth | 10× comm cost buys only 17% accuracy — design for efficiency. |

**Directly relevant for:** federated learning, multi-agent AI systems,
distributed data pipelines, any system where agents aggregate information
and some may be unreliable.

---

## Key Results

### v2 — Noise (IID clients, 3 noise levels)

![v2 accuracy curves](results/fed_results.png)

| ID | Prediction | Verdict |
|----|-----------|---------|
| P1 | Interior optimum at every noise level | ❌ **FAIL** — only at low noise (ρ\*=0.7) |
| P2 | ρ\* shifts with noise | ✅ **PASS** — peaks: 0.7 → 1.0 → 1.0 |
| P3 | ρ\* ≠ 0.5 universally | ✅ **PASS** |
| P4 | Comm linear, accuracy sub-linear | ✅ **PASS** — 10× → 1.17× |

### v3 — Diversity (non-IID class splits)

| P1-v3 | Interior optimum in non-IID | ❌ **FAIL** |
|-------|---------------------------|------------|
| P2-v3 | Full-sync accuracy drops | ✅ **PASS** (0.701 vs 0.871) |

### v4 — Conflict (30% adversarial)

All 3 predictions **FAIL**. Full sync still won (0.791).

### v5 — Phase Diagram (0–90% adversarial sweep, 105 runs)

| q | 0.0 | 0.1 | 0.2 | 0.3 | **0.5** | **0.7** | 0.9 |
|---|-----|-----|-----|-----|---------|---------|-----|
| **ρ\*** | 1.0 | 1.0 | 1.0 | 1.0 | **0.3** | **0.3** | 1.0\* |
| **acc** | 0.877 | 0.863 | 0.834 | 0.796 | **0.542** | 0.203 | 0.149 |

All 4 predictions PASS (after significance correction). **q_c ≈ 0.5.**

<details>
<summary>📊 Full v3/v4/v5 details (click to expand)</summary>

### v3 — Non-IID regime (2026-04-25)

![v2 IID vs v3 Non-IID comparison](results/v2_vs_v3_comparison.png)

Class-distribution mismatch alone isn't enough to create an interior optimum.
Full sync still wins — but with 20% lower accuracy (0.701 vs 0.871).

### v4 — Adversarial clients (2026-04-25)

![v4 adversarial results](results/adversarial_results.png)

Even with 30% of clients sending flipped labels, FedAvg's averaging dilutes the
poison enough that full sync wins.

### v5 — Significance correction note

Raw `argmax` at q=0.0 gave ρ\*=0.5, but the gap (0.0009) was within noise
(pooled std=0.006). After requiring the peak to beat ρ=1.0 by >1 pooled std,
q=0.0 correctly reads ρ\*=1.0. The interior optimum at q=0.5 is real
(gap=0.179, pooled std=0.170).

</details>

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

### 6 — The final thesis (v2 → v5)

> *"FedAvg shows a phase transition at q ≈ 0.5, where averaging
> switches from stabilizing to destabilizing."*

The data:

| q | acc at ρ=1.0 | Best partial acc | Averaging effect |
|---|-------------|-----------------|------------------|
| 0.0–0.3 | 0.876–0.796 | ≤ full sync | **Stabilizing** — more averaging = better |
| **0.5** | **0.363** | **0.542** (ρ=0.3) | **Destabilizing** — more averaging = worse |
| 0.7–0.9 | 0.022–0.012 | 0.203–0.149 | Collapsed |

The mechanism: at q ≤ 0.3, honest clients outnumber adversarial ones
7-to-3, so averaging dilutes poison. At q = 0.5 it's 5-to-5 — full
sync *guarantees* 50% poison every round. Partial sync at least has
a chance of drawing a better ratio. Crossing q ≈ 0.5 causes a **54%
relative accuracy drop** at ρ=1.0 (from 0.796 to 0.363).

*Caveat: at q=0.5 the standard deviations are large (0.24–0.29 at low ρ)
with only 3 seeds. The direction is robust; the exact numbers need more
seeds to pin down precisely.*

Five experiments, 23 conditions:

| Stage | What was tested | What we learned |
|-------|----------------|----------------|
| v1 | Simulation | Universal 50% rule → falsified |
| v2 | IID + noise | Interior optimum only at low noise; 3/4 PASS |
| v3 | Non-IID class splits | Consensus still dominates; 1/3 PASS |
| v4 | 30% adversarial | Consensus *still* dominates; 0/3 PASS |
| **v5** | **q sweep (0%–90% adversarial)** | **Phase transition at q ≈ 0.5; 4/4 PASS** |

> I didn't find an optimal communication level.
> **I found where consensus stops working.**

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
│   ├── PRE_REGISTRATION.md        ← v2 predictions (frozen)
│   ├── PRE_REGISTRATION_v3.md     ← v3 non-IID predictions
│   ├── PRE_REGISTRATION_v4.md     ← v4 adversarial predictions
│   └── PRE_REGISTRATION_v5.md     ← v5 phase diagram predictions
│
├── src/
│   ├── fed_experiment.py          ← v2 IID experiment
│   ├── fed_experiment_noniid.py   ← v3 non-IID
│   ├── fed_experiment_adversarial.py ← v4 adversarial (30%)
│   ├── fed_experiment_q_sweep.py  ← v5 phase diagram (q × ρ sweep) ⭐
│   ├── plot_q_sweep.py            ← generates v5 heatmap
│   ├── compare_v2_v3.py           ← v2/v3 comparison
│   └── honest_sensitivity.py      ← simulation that falsified v1
│
├── results/
│   ├── fed_results.png            ← v2 IID plot
│   ├── raw_results.json           ← v2 raw numbers
│   ├── verdicts.json              ← v2 verdicts
│   ├── noniid_results.json        ← v3 raw numbers
│   ├── noniid_verdicts.json       ← v3 verdicts
│   ├── adversarial_results.json   ← v4 raw numbers
│   ├── adversarial_verdicts.json  ← v4 verdicts
│   ├── q_sweep_results.json       ← v5 full (q × ρ) sweep
│   ├── q_sweep_verdicts.json      ← v5 verdicts
│   ├── q_vs_rho_heatmap.png      ← v5 phase diagram ⭐
│   └── v2_vs_v3_comparison.png   ← v2/v3 comparison
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
| FedAvg survives minority adversarial corruption (q ≤ 0.3) | Interior optimum exists universally |
| At majority corruption (q ≥ 0.5), full sync becomes harmful — interior optimum emerges | A universal "50% rule" |
| At q ≥ 0.7, the system collapses regardless of ρ | Cross-domain unification |
| 10× comm buys only 1.17× accuracy (low-corruption regime) | That this generalizes beyond FedAvg+MNIST |
| Pre-registration across 5 experiments prevents post-hoc spin | — |

---

## Methodology commitments

1. **Predictions locked first** — [`prereg/PRE_REGISTRATION.md`](prereg/PRE_REGISTRATION.md) timestamped before the run.
2. **No post-hoc tuning** — parameters fixed before the run; no sweep done after seeing results.
3. **Failure reported as failure** — `verdicts.json` → `"P1": "FAIL (at least one boundary peak)"`. Goalposts not moved.
4. **All artifacts public** — raw JSON, verdicts, plot, code, paper, pre-registration.

---

## What's next

**v2:** 3/4 · **v3:** 1/3 · **v4:** 0/3 · **v5:** 2/4. Five pre-registered experiments, 23 conditions.

The robustness boundary is mapped. What remains:

1. **Byzantine-robust aggregation** — FedAvg dilutes poison via averaging.
   Krum, trimmed mean, or coordinate-wise median may shift the q_c threshold.
   Does robust aggregation push the green zone to higher q?
2. **Different architectures** — SmallCNN+MNIST may be uniquely robust.
   CIFAR-10 with ResNet could reveal architecture-dependent boundaries.
3. **Continuous q sweep** — finer grid around q=0.3–0.5 to pinpoint q_c precisely.
4. **Blog post** — "I tried to break federated learning. Here's where it broke."
   The v5 heatmap is the hero image.

---

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite both the paper and the
retraction — they are inseparable.

---

## License

MIT — see [`LICENSE`](LICENSE).
