"""
v2 IID vs v3 Non-IID Comparison Plot
=====================================
Loads both result sets and produces a side-by-side comparison
showing what changed when we moved from IID to non-IID clients.
"""

import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── paths ──────────────────────────────────────────────────
BASE = pathlib.Path(__file__).resolve().parent.parent / "results"
v2_path  = BASE / "raw_results.json"
v3_path  = BASE / "noniid_results.json"
out_path = BASE / "v2_vs_v3_comparison.png"

with open(v2_path)  as f: v2 = json.load(f)
with open(v3_path)  as f: v3_raw = json.load(f)

RHOS = [0.1, 0.3, 0.5, 0.7, 1.0]

# v3 results may be a list of dicts or a dict keyed by rho-string
if isinstance(v3_raw["results"], list):
    v3 = {r["rho"]: {"mean": r["acc_mean"], "std": r["acc_std"]} for r in v3_raw["results"]}
else:
    v3 = {float(k): v for k, v in v3_raw["results"].items()}

# ── colors ──────────────────────────────────────────────────
C_LOW  = "#2196F3"
C_MED  = "#FF9800"
C_HIGH = "#F44336"
C_V3   = "#9C27B0"

# ── figure ──────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32)

fig.suptitle(
    "v2 (IID) vs v3 (Non-IID) — Tracking the Interior Optimum\n"
    "Pre-registered experiment · honest reporting",
    fontsize=13, fontweight="bold",
)

# ── Panel A: v2 IID curves ───────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
colors = {"low": C_LOW, "medium": C_MED, "high": C_HIGH}
labels = {"low": "IID low noise (5%)", "medium": "IID medium noise (20%)", "high": "IID high noise (40%)"}
for noise in ["low", "medium", "high"]:
    accs = [r["acc_mean"] for r in v2[noise]]
    stds = [r["acc_std"]  for r in v2[noise]]
    peak = int(np.argmax(accs))
    ax_a.errorbar(RHOS, accs, yerr=stds, marker="o", linewidth=2,
                  capsize=3, color=colors[noise], label=labels[noise])
    ax_a.annotate(f"ρ*={RHOS[peak]}",
                  xy=(RHOS[peak], accs[peak]),
                  xytext=(RHOS[peak] - 0.13, accs[peak] + 0.006),
                  fontsize=8, color=colors[noise], fontweight="bold")

ax_a.axvspan(0.1, 0.9, alpha=0.06, color="green", label="Interior region")
ax_a.set_title("Panel A · v2  IID Clients", fontsize=11, fontweight="bold")
ax_a.set_xlabel("Sync density ρ"); ax_a.set_ylabel("Test accuracy")
ax_a.legend(fontsize=8); ax_a.grid(alpha=0.3); ax_a.set_ylim(0.60, 0.92)

# ── Panel B: v3 Non-IID curve ────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
v3_means = [v3[r]["mean"] for r in RHOS]
v3_stds  = [v3[r]["std"]  for r in RHOS]
ax_b.errorbar(RHOS, v3_means, yerr=v3_stds, marker="s", linewidth=2.5,
              capsize=3, color=C_V3, label="Non-IID (2 classes/client)")
ax_b.axvspan(0.1, 0.9, alpha=0.06, color="green", label="Interior region")
ax_b.annotate("[FAIL] Still\nboundary peak", xy=(1.0, v3_means[-1]),
              xytext=(0.55, v3_means[-1] - 0.08),
              fontsize=9, color=C_V3, fontweight="bold",
              arrowprops=dict(arrowstyle="->", color=C_V3))
ax_b.set_title("Panel B · v3  Non-IID Clients", fontsize=11, fontweight="bold")
ax_b.set_xlabel("Sync density ρ"); ax_b.set_ylabel("Test accuracy")
ax_b.legend(fontsize=9); ax_b.grid(alpha=0.3); ax_b.set_ylim(0.10, 0.80)

# ── Panel C: Full-sync accuracy comparison ───────────────────
ax_c = fig.add_subplot(gs[1, 0])
v2_fullsync = {
    "low":    v2["low"][-1]["acc_mean"],
    "medium": v2["medium"][-1]["acc_mean"],
    "high":   v2["high"][-1]["acc_mean"],
}
v3_fullsync = v3[1.0]["mean"]

bar_labels = ["IID low", "IID medium", "IID high", "Non-IID"]
bar_vals   = [v2_fullsync["low"], v2_fullsync["medium"], v2_fullsync["high"], v3_fullsync]
bar_colors = [C_LOW, C_MED, C_HIGH, C_V3]

bars = ax_c.bar(bar_labels, bar_vals, color=bar_colors, edgecolor="black", linewidth=0.8)
for bar, val in zip(bars, bar_vals):
    ax_c.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
              f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax_c.axhline(v3_fullsync, color=C_V3, linestyle="--", linewidth=1, alpha=0.5)
ax_c.set_title("Panel C · Full Sync (rho=1.0) Accuracy\nNon-IID dramatically lower  [P2-v3 PASS]",
               fontsize=10, fontweight="bold")
ax_c.set_ylabel("Test accuracy"); ax_c.set_ylim(0, 1.0)
ax_c.grid(axis="y", alpha=0.3)

# ── Panel D: Slope comparison (how fast accuracy rises with ρ) ─
ax_d = fig.add_subplot(gs[1, 1])
# Normalize each series to [0,1] to compare shape
def norm(vals):
    lo, hi = min(vals), max(vals)
    return [(v - lo) / (hi - lo) if hi > lo else 0.0 for v in vals]

for noise, color in colors.items():
    accs = [r["acc_mean"] for r in v2[noise]]
    ax_d.plot(RHOS, norm(accs), marker="o", linewidth=1.8,
              color=color, alpha=0.7, label=f"IID {noise}")
ax_d.plot(RHOS, norm(v3_means), marker="s", linewidth=2.5,
          color=C_V3, label="Non-IID")

ax_d.set_title("Panel D · Normalized Gain Shape\n(shows where each curve bends)",
               fontsize=10, fontweight="bold")
ax_d.set_xlabel("Sync density ρ"); ax_d.set_ylabel("Normalized accuracy gain")
ax_d.legend(fontsize=8); ax_d.grid(alpha=0.3)

plt.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"✅ Comparison plot saved: {out_path}")

# ── Print the honest verdict summary ────────────────────────────
print()
print("=" * 60)
print("V3 VERDICT SUMMARY  (non-IID regime test)")
print("=" * 60)
verdicts_v3 = {
    "P1-v3": ("Interior optimum ρ* ∈ (0,1) in non-IID regime",
               "FAIL", f"ρ*=1.0 (boundary) — P1 fully falsified across both regimes"),
    "P2-v3": ("Full-sync accuracy lower in non-IID than IID",
               "PASS", f"Non-IID ρ=1.0 → {v3_fullsync:.3f} vs IID low → {v2_fullsync['low']:.3f}"),
    "P3-v3": ("Non-IID peak ρ* shifts left vs IID high-noise",
               "FAIL", "ρ*=1.0 in both cases — no leftward shift"),
}
for pid, (pred, verdict, evidence) in verdicts_v3.items():
    sym = "✅" if verdict == "PASS" else "❌"
    print(f"  {sym} {pid}  [{verdict}]")
    print(f"     Prediction: {pred}")
    print(f"     Evidence:   {evidence}")
    print()

print("─" * 60)
print("NET: The diversity-penalty story needs a harder test.")
print("     Non-IID class splits were not enough to shrink β/γ.")
print("     Next step: adversarial clients or constrained compute.")
print("─" * 60)
