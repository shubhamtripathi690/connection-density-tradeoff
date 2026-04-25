"""Re-analyze v5 data with significance-corrected verdict logic."""
import json
import numpy as np

with open("../results/q_sweep_results.json") as f:
    data = json.load(f)

results = data["results"]
rhos = [0.1, 0.3, 0.5, 0.7, 1.0]
q_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

lookup = {(r["q"], r["rho"]): r["acc_mean"] for r in results}
lookup_std = {(r["q"], r["rho"]): r["acc_std"] for r in results}

print("=" * 70)
print("V5 RE-ANALYSIS (significance-corrected)")
print("Rule: interior peak must beat rho=1.0 by > 1 pooled std")
print("=" * 70)

print(f"\n  {'q':>4}  {'raw':>5}  {'acc@pk':>8}  {'acc@1.0':>8}  "
      f"{'gap':>7}  {'p.std':>7}  {'corrected':>10}")
print("  " + "-" * 65)

peak_rhos = {}
for q in q_values:
    accs = {rho: lookup[(q, rho)] for rho in rhos}
    raw_best = max(accs, key=accs.get)
    raw_idx = rhos.index(raw_best)
    acc_peak = accs[raw_best]
    acc_full = accs[1.0]
    s1 = lookup_std[(q, raw_best)]
    s2 = lookup_std[(q, 1.0)]
    pooled = np.sqrt((s1**2 + s2**2) / 2)
    gap = acc_peak - acc_full

    if 0 < raw_idx < len(rhos) - 1 and gap > pooled:
        rho_star = raw_best
        label = f"rho*={rho_star} SIG"
    else:
        rho_star = 1.0
        label = "rho*=1.0" + (" (noise)" if raw_best != 1.0 else "")

    peak_rhos[q] = rho_star
    print(f"  {q:>4.1f}  {raw_best:>5.1f}  {acc_peak:>8.4f}  {acc_full:>8.4f}  "
          f"{gap:>+7.4f}  {pooled:>7.4f}  {label}")

# Verdicts
print("\n" + "-" * 70)

q_c = None
for q in q_values:
    if peak_rhos[q] < 1.0:
        q_c = q
        break

if q_c is not None:
    print(f"  P1-v5 PASS: q_c = {q_c}")
else:
    print(f"  P1-v5 FAIL: full sync wins at every q")

# P2
acc_05_full = lookup[(0.5, 1.0)]
best_partial = max(lookup[(0.5, rho)] for rho in rhos if rho < 1.0)
p2 = acc_05_full < best_partial
print(f"  P2-v5 {'PASS' if p2 else 'FAIL'}: q=0.5 full={acc_05_full:.4f} vs best_partial={best_partial:.4f}")

# P3
best_07 = max(lookup[(0.7, rho)] for rho in rhos)
p3 = best_07 < 0.30
print(f"  P3-v5 {'PASS' if p3 else 'FAIL'}: q=0.7 best={best_07:.4f} ({'<' if p3 else '>='} 0.30)")

# P4
p4 = all(peak_rhos[q] == 1.0 for q in q_values if q < 0.3)
print(f"  P4-v5 {'PASS' if p4 else 'FAIL'}: rho*=1.0 for all q<0.3")

print(f"\n  Corrected peak rhos: {peak_rhos}")
print(f"  Corrected q_c = {q_c}")

# Save corrected verdicts
verdicts = {
    "experiment": "adversarial fraction sweep v5.0 (corrected)",
    "significance_rule": "interior peak must beat rho=1.0 by >1 pooled std",
    "q_c": q_c,
    "peak_rhos": {str(q): peak_rhos[q] for q in q_values},
    "P1-v5": "PASS" if q_c else "FAIL",
    "P2-v5": "PASS" if p2 else "FAIL",
    "P3-v5": "PASS" if p3 else "FAIL",
    "P4-v5": "PASS" if p4 else "FAIL",
}
with open("../results/q_sweep_verdicts.json", "w") as f:
    json.dump(verdicts, f, indent=2)
print(f"\n  Saved: ../results/q_sweep_verdicts.json")
