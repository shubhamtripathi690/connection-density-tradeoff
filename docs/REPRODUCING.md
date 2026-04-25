# Reproducing the Results

This guide reproduces the v2.1 numbers exactly (modulo PyTorch
nondeterminism on different hardware).

---

## 1. Prerequisites

- Python 3.11 (other 3.10+ versions likely work but are not pinned)
- ~2 GB free disk space (MNIST + venv)
- ~10 minutes on a recent CPU; ~2 minutes on a GPU

---

## 2. Setup

```bash
git clone <this-repo-url>
cd connection-density-tradeoff

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Run the federated experiment

```bash
cd src
python fed_experiment.py
```

Expected console output (truncated):

```
Using device: cpu
================================================================
FEDERATED LEARNING — PRE-REGISTERED EXPERIMENT
Testing predictions P1–P4 from PRE_REGISTRATION.md
================================================================
...
  ρ=0.1  acc=0.7430±0.0897  comm=10  time=...
  ρ=0.3  acc=0.8341±0.0224  comm=30  time=...
  ...
================================================================
FINAL VERDICT (Pre-Registered)
================================================================
  ❌ P1: FAIL (at least one boundary peak)
  ✅ P2: PASS
  ✅ P3: PASS
  ✅ P4: PASS
```

Outputs are written to `../results/`:

- `raw_results.json` — per-(noise, ρ) accuracy means and stds
- `verdicts.json` — P1–P4 pass/fail
- `fed_results.png` — accuracy + comm-cost plots

---

## 4. Run the sensitivity simulation (optional)

```bash
cd src
python honest_sensitivity.py
```

This generates the 36-regime sweep that originally falsified the
"universal 50%" claim. Output: `honest_sensitivity.png`.

---

## 5. Determinism notes

- The numpy seed is fixed (`SEED = 42`).
- The torch CPU seed is fixed.
- Per-rho results are averaged over 3 seeds (0, 1, 2) inside
  `federated_run`.
- Exact accuracy may vary by ~0.001–0.005 across PyTorch versions
  and BLAS backends. The verdicts (P1–P4) are stable.

---

## 6. If a verdict changes

If you re-run on different hardware and a verdict flips:

1. **Do not** edit the pre-registration to match the new verdict.
2. **Do** open an issue / PR with your full output and environment.
3. The v2.1 numbers in [paper/RESULTS_REPORT.md](../paper/RESULTS_REPORT.md)
   are the authoritative ones for that release.

A v2.2+ release would document any verified, hardware-independent
shift.
