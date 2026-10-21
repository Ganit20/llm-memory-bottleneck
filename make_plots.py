"""Builds the quality-vs-SSD-traffic trade-off plot and a markdown table from results/exp4*.json.

Usage: python make_plots.py   ->  results/tradeoff.png, results/summary.md
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
# Throughput model from sim.py: Qwen3-32B @4-bit (19 GB, FFN 13 GB) on a 16 GB Mac, 200 GB/s RAM, 5 GB/s SSD
REST, FFN, RAM_BW, SSD_BW = 6e9, 13e9, 200e9, 5e9


def tok_s(frac, read):
    return 1 / max((REST + frac * FFN) / RAM_BW, read * FFN / SSD_BW)


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else {}


base30, f05 = load("exp4.json"), load("exp4_f0.5.json")
ctrl = base30["ctrl"]["dense"]
points = []   # (label, frac, lambda, ppl, read)
for frac, src in [(0.3, base30), (0.5, f05)]:
    if "r_look" in src:
        points.append(("no reg.", frac, 0.0, src["r_look"]["router_group"], src["r_look"]["router_cache"]["ffn_read_per_token"]))
    if "r_look_tl" in src:
        points.append(("λ=0.5", frac, 0.5, src["r_look_tl"]["router_group"], src["r_look_tl"]["router_cache"]["ffn_read_per_token"]))
for lam in [1.0, 2.0]:
    d = load(f"exp4_f0.5_tl{lam}.json").get("r_look_tl")
    if d:
        points.append((f"λ={lam:g}", 0.5, lam, d["router_group"], d["router_cache"]["ffn_read_per_token"]))

fig, ax = plt.subplots(figsize=(7, 4.5))
for frac, marker in [(0.3, "o"), (0.5, "s")]:
    pts = sorted([p for p in points if p[1] == frac], key=lambda p: p[2])
    ax.plot([p[4] * 100 for p in pts], [(p[3] / ctrl - 1) * 100 for p in pts], marker=marker, label=f"{int(frac * 100)}% of FFN blocks active")
    for p in pts:
        ax.annotate(p[0], (p[4] * 100, (p[3] / ctrl - 1) * 100), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.axvline(SSD_BW / 5 / FFN * 100, ls="--", c="gray", lw=1)
ax.text(SSD_BW / 5 / FFN * 100, ax.get_ylim()[1] * 0.97, " 5 tok/s budget", fontsize=8, c="gray", va="top")
ax.set_xlabel("FFN weights read from SSD per token (%)  ← better")
ax.set_ylabel("Perplexity increase vs dense fine-tune (%)  ← better")
ax.set_title("Qwen3-0.6B, lookahead router + temporal-locality loss")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(R / "tradeoff.png", dpi=150)

rows = ["| active blocks | locality weight λ | PPL (dense ctrl = %.1f) | PPL increase | SSD reads / token | est. tok/s (32B @ 16 GB) |" % ctrl,
        "|---|---|---|---|---|---|"]
for lab, frac, lam, ppl, read in sorted(points, key=lambda p: (p[1], p[2])):
    rows.append(f"| {int(frac * 100)}% | {lam:g} | {ppl:.1f} | +{(ppl / ctrl - 1) * 100:.0f}% | {read * 100:.1f}% | {tok_s(frac, read):.1f} |")
(R / "summary.md").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
