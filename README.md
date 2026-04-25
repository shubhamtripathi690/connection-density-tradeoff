# Connection-Density Trade-offs in Multi-Agent Systems

> A pre-registered, honestly-reported empirical study of how
> communication density (ρ) affects performance in noisy multi-agent
> systems — applied to federated learning on MNIST.

[![status](https://img.shields.io/badge/status-working--paper-blue)]()
[![pre--registered](https://img.shields.io/badge/pre--registered-yes-brightgreen)]()
[![results](https://img.shields.io/badge/results-3%2F4%20PASS%2C%201%2F4%20FAIL-orange)]()
[![license](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## TL;DR

I pre-registered four predictions about a multi-agent connection-density
trade-off model **before** running any real experiment. Then I ran the
experiment on MNIST federated learning. **Three predictions held; one
failed.** This repo contains the code, raw results, paper, and the
honest retraction.

| ID | Prediction | Verdict |
|----|-----------|---------|
| **P1** | Interior optimum ρ\* ∈ (0,1) for every noise level | ❌ **FAIL** (only at low noise) |
| **P2** | ρ\* shifts with client noise | ✅ **PASS** (peaks: 0.7 → 1.0 → 1.0) |
| **P3** | ρ\* ≠ 0.5 universally | ✅ **PASS** (no peak at 0.5) |
| **P4** | Comm cost linear, accuracy gain sub-linear | ✅ **PASS** (10× cost → 1.17× accuracy) |

> A 4/4 pass would have been suspicious. A 0/4 fail would have been a
> refutation. **3/4 with one honest retraction is what calibrated
> science looks like.**

---

## Why this repo exists

A previous version of this framework claimed a *universal* `ρ* = 0.5`
rule across domains. That claim was self-falsified via a sensitivity
analysis. This repo is the **next step**: take what survived, pre-register
real predictions, run a real experiment, and report the result —
**including the failure** — without moving the goalposts.

---

## Repository layout

```
connection-density-tradeoff/
├── README.md                     ← you are here
├── LICENSE                       ← MIT
├── CITATION.cff                  ← how to cite
├── CHANGELOG.md                  ← version history
├── requirements.txt              ← pinned Python deps
├── .gitignore
│
├── paper/
│   ├── HONEST_PAPER.md           ← the paper itself (v2.1)
│   ├── RESULTS_REPORT.md         ← detailed results writeup
│   └── TRUTH_FALSITY_MATRIX.md   ← what survived, what didn't
│
├── prereg/
│   └── PRE_REGISTRATION.md       ← locked predictions (timestamped)
│
├── src/
│   ├── fed_experiment.py         ← the federated MNIST experiment
│   └── honest_sensitivity.py     ← the simulation that falsified v1
│
├── results/
│   ├── raw_results.json          ← per-(noise, ρ) accuracies
│   ├── verdicts.json             ← machine-readable P1–P4 verdicts
│   └── fed_results.png           ← accuracy + comm-cost plots
│
└── docs/
    └── REPRODUCING.md            ← step-by-step reproduction guide
```

---

## Quick start

### Reproduce the experiment

```bash
git clone <this-repo>
cd connection-density-tradeoff

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd src
python fed_experiment.py
```

Runs in ~5–10 minutes on CPU. Outputs are written to `../results/`.

### Reproduce the simulation

```bash
cd src
python honest_sensitivity.py
```

Generates the 36-regime sensitivity sweep that falsified the original
"universal 50%" claim.

---

## Headline numbers

Test accuracy (mean over 3 seeds, 10 federated rounds, 10 clients):

| Noise | ρ=0.1 | ρ=0.3 | ρ=0.5 | ρ=0.7 | ρ=1.0 | Peak |
|-------|-------|-------|-------|-------|-------|------|
| low (5%)    | 0.743 | 0.834 | 0.853 | **0.876** | 0.871 | ρ=0.7 |
| medium (20%)| 0.669 | 0.799 | 0.841 | 0.853 | **0.866** | ρ=1.0 |
| high (40%)  | 0.674 | 0.764 | 0.779 | 0.770 | **0.839** | ρ=1.0 |

Communication cost is exactly linear in ρ as predicted: 10, 30, 50, 70, 100.

See [paper/RESULTS_REPORT.md](paper/RESULTS_REPORT.md) for the full
analysis and the honest interpretation of why P1 failed.

---

## What this repo claims (and does not claim)

**Does claim:**
- ✅ Multi-agent systems with competing consensus/diversity pressures
  *can* have an interior optimum in connection density
- ✅ The optimum's location is **system-specific**, not universal
- ✅ For practical federated learning, communication cost grows much
  faster than accuracy gain (10× → 1.17×)

**Does not claim:**
- ❌ A universal "50% rule" of any kind
- ❌ That the framework unifies quantum / social / biological systems
- ❌ That an interior optimum exists at every noise level (P1 was
  falsified — this is acknowledged, not hidden)

---

## Methodology commitments

1. **Pre-registration first.** Predictions in [prereg/PRE_REGISTRATION.md](prereg/PRE_REGISTRATION.md)
   were locked before the experiment ran.
2. **No post-hoc tuning.** Hyperparameters and noise levels were fixed
   before the run; no sweep was done to make P1 pass.
3. **Failure is reported as failure.** P1's verdict in
   [results/verdicts.json](results/verdicts.json) is `FAIL`. The paper
   contains an explicit retraction in §6.
4. **All artifacts published.** Raw JSON, verdicts, plots, code, paper,
   and pre-registration all live in this repo.

---

## What's next

1. **Test the regime-correction hypothesis.** Run with non-IID client
   splits (each client only sees 2 digit classes). The diversity
   penalty γ should be much larger there. If an interior optimum
   appears, P1 is rehabilitated under a *scoped* form. If it doesn't,
   that's another honest data point.
2. Adversarial-client experiments (some clients deliberately wrong).
3. Larger-scale runs with more clients and more rounds.

These will land as separate, pre-registered v3.0 experiments.

---

## Citation

See [CITATION.cff](CITATION.cff). If you cite this work, please cite
**both** the paper *and* the honest retraction — they're a package.

---

## License

MIT — see [LICENSE](LICENSE).
