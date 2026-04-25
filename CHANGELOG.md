# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.1] — 2026-04-25 — Post-empirical revision

### Added
- Federated learning experiment on MNIST (`src/fed_experiment.py`)
- Raw results (`results/raw_results.json`)
- Verdicts (`results/verdicts.json`)
- Plot (`results/fed_results.png`)
- `paper/RESULTS_REPORT.md` — detailed empirical writeup
- `paper/TRUTH_FALSITY_MATRIX.md` — what survived contact with data
- §6 retraction in `paper/HONEST_PAPER.md`
- §4.1 real-data row in the sensitivity table

### Changed
- P1 ("interior optimum exists for every noise level") **scoped down**
  to "may exist when β/γ is moderate"
- Practical guidance in §7 of the paper sharpened with regime advice

### Verdicts (pre-registered)
- P1: ❌ FAIL — only the low-noise regime had an interior peak
- P2: ✅ PASS — peaks shifted (0.7 → 1.0 → 1.0) with noise
- P3: ✅ PASS — no noise level peaked at ρ = 0.5
- P4: ✅ PASS — 10× comm cost yielded only 1.17× accuracy

### Not changed (deliberately)
- The pre-registration in `prereg/PRE_REGISTRATION.md` is **frozen**.
  It is not edited to make P1 pass.

---

## [2.0] — 2026-04-25 — Honest revision (pre-experiment)

### Added
- `src/honest_sensitivity.py` — 36-regime simulation
- Truth-falsity matrix v1
- Pre-registration document

### Changed
- Replaced the v1.x "universal ρ\* = 0.5" claim with a parameterized
  framework: `ρ\* = f(σ², η, p, graph spectrum)`
- Added explicit scope and limitations section
- Added "disconfirming evidence acknowledged" section

### Removed
- Cosmic-web / cross-domain unification claims
- Any suggestion that ρ\* = 0.5 is a constant of nature

---

## [1.x] — pre-history

Earlier drafts claimed a universal "50% rule" across domains.
Self-falsified by the v2.0 sensitivity analysis. Not reproduced here.
