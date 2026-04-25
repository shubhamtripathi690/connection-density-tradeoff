"""
Phase Diagram: FedAvg Robustness Map (v5.0)
============================================
Generates a heatmap of accuracy vs (q, ρ) — shows exactly where
FedAvg consensus breaks under adversarial corruption.

Run: python plot_q_sweep.py
Requires: ../results/q_sweep_results.json (from fed_experiment_q_sweep.py)
"""

import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

BASE = pathlib.Path(__file__).resolve().parent.parent / "results"

with open(BASE / "q_sweep_results.json") as f:
    data = json.load(f)

results = data["results"]
q_vals = sorted(set(r["q"] for r in results))
rho_vals = sorted(set(r["rho"] for r in results))

# Build accuracy matrix (q rows × ρ columns)
acc_matrix = np.zeros((len(q_vals), len(rho_vals)))
std_matrix = np.zeros_like(acc_matrix)
for r in results:
    i = q_vals.index(r["q"])
    j = rho_vals.index(r["rho"])
    acc_matrix[i, j] = r["acc_mean"]
    std_matrix[i, j] = r["acc_std"]

# Find ρ* for each q WITH significance check
# Interior peak must beat ρ=1.0 by > 1 pooled std
peak_rhos = []
for i, q in enumerate(q_vals):
    j_raw = int(np.argmax(acc_matrix[i]))
    j_full = rho_vals.index(1.0)
    acc_peak = acc_matrix[i, j_raw]
    acc_full = acc_matrix[i, j_full]
    pooled = np.sqrt((std_matrix[i, j_raw]**2 + std_matrix[i, j_full]**2) / 2)
    gap = acc_peak - acc_full
    if 0 < j_raw < len(rho_vals) - 1 and gap > pooled:
        peak_rhos.append(rho_vals[j_raw])  # significant interior peak
    else:
        peak_rhos.append(1.0)  # default to full sync

# ── Figure: 3 panels ───────────────────────────────────────
fig = plt.figure(figsize=(18, 6))
fig.suptitle("v5 — FedAvg Robustness Phase Diagram\n"
             "Where does consensus break under adversarial corruption?",
             fontsize=14, fontweight="bold")

# Panel A: Heatmap
ax1 = fig.add_subplot(131)
im = ax1.imshow(acc_matrix, aspect="auto", origin="lower",
                cmap="RdYlGn", vmin=0.0, vmax=1.0)
ax1.set_xticks(range(len(rho_vals)))
ax1.set_xticklabels([f"{r:.1f}" for r in rho_vals])
ax1.set_yticks(range(len(q_vals)))
ax1.set_yticklabels([f"{q:.1f}" for q in q_vals])
ax1.set_xlabel("Sync density (rho)", fontsize=11)
ax1.set_ylabel("Adversarial fraction (q)", fontsize=11)
ax1.set_title("Panel A: Accuracy Heatmap", fontsize=11, fontweight="bold")

# Annotate each cell
for i in range(len(q_vals)):
    for j in range(len(rho_vals)):
        val = acc_matrix[i, j]
        color = "white" if val < 0.4 else "black"
        ax1.text(j, i, f"{val:.2f}", ha="center", va="center",
                 fontsize=8, color=color, fontweight="bold")

# Mark ρ* per row
for i, (q, rho_star) in enumerate(zip(q_vals, peak_rhos)):
    j = rho_vals.index(rho_star)
    ax1.scatter(j, i, marker="*", s=200, color="black", zorder=5,
                edgecolor="white", linewidth=0.8)

cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label("Test accuracy", fontsize=10)

# Panel B: Accuracy curves per q
ax2 = fig.add_subplot(132)
cmap = plt.cm.coolwarm
for i, q in enumerate(q_vals):
    color = cmap(i / (len(q_vals) - 1))
    accs = acc_matrix[i]
    stds = std_matrix[i]
    ax2.errorbar(rho_vals, accs, yerr=stds, marker="o", linewidth=1.8,
                 capsize=3, color=color, label=f"q={q:.1f}")
    # Mark peak
    j_star = int(np.argmax(accs))
    ax2.scatter([rho_vals[j_star]], [accs[j_star]], s=80, zorder=5,
                color=color, edgecolor="black", linewidth=0.8)

ax2.set_xlabel("Sync density (rho)", fontsize=11)
ax2.set_ylabel("Test accuracy", fontsize=11)
ax2.set_title("Panel B: Accuracy vs rho\n(one curve per q)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=8, ncol=2, loc="lower right")
ax2.grid(alpha=0.3)
ax2.set_ylim(-0.05, 1.0)

# Panel C: ρ* vs q (the critical threshold chart)
ax3 = fig.add_subplot(133)
ax3.plot(q_vals, peak_rhos, marker="s", linewidth=2.5, color="#E91E63",
         markersize=10, zorder=3)
ax3.axhline(1.0, color="grey", linestyle="--", alpha=0.5, label="Full sync (rho=1.0)")
ax3.fill_between(q_vals, 0, 1, alpha=0.05, color="green")

# Find q_c
q_c = None
for q, rho_star in zip(q_vals, peak_rhos):
    if rho_star < 1.0:
        q_c = q
        break

if q_c is not None:
    ax3.axvline(q_c, color="red", linestyle="-.", linewidth=2, alpha=0.7,
                label=f"q_c = {q_c} (consensus breaks)")
    ax3.annotate(f"q_c = {q_c}\nConsensus breaks here",
                 xy=(q_c, peak_rhos[q_vals.index(q_c)]),
                 xytext=(q_c + 0.1, 0.5),
                 fontsize=10, fontweight="bold", color="red",
                 arrowprops=dict(arrowstyle="->", color="red", lw=2))

ax3.set_xlabel("Adversarial fraction (q)", fontsize=11)
ax3.set_ylabel("Optimal rho*", fontsize=11)
ax3.set_title("Panel C: Where Does Full Sync\nStop Being Optimal?", fontsize=11, fontweight="bold")
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)
ax3.set_xlim(-0.05, 0.95)
ax3.set_ylim(-0.05, 1.15)

plt.tight_layout()
out_path = BASE / "q_vs_rho_heatmap.png"
plt.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"✅ Phase diagram saved: {out_path}")

# Print summary
print(f"\nPeak ρ* per adversarial fraction:")
for q, rho_star in zip(q_vals, peak_rhos):
    marker = " ← consensus breaks" if rho_star < 1.0 else ""
    print(f"  q={q:.1f}  →  ρ*={rho_star:.1f}{marker}")
if q_c:
    print(f"\n🔥 Critical adversarial fraction: q_c = {q_c}")
else:
    print(f"\n   Full sync wins at every q level tested.")
