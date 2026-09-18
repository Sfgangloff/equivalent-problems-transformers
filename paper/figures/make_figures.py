"""Generates Figures 1 and 2 for the paper from results/*.jsonl.

Run from the repo root:
    source .venv/bin/activate && python paper/figures/make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load(fname):
    path = os.path.join(RESULTS, fname)
    with open(path) as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# Figure 1: mixed-alphabet per-trial cell accuracy across four conditions.
# ---------------------------------------------------------------------------
conditions_fig1 = [
    ("Classifier\nK=6, fixed", "eval_mixed_results.jsonl"),
    ("Pointer\nK=6, fixed", "eval_pointer_mixed_results.jsonl"),
    ("Pointer\nK=25, mixed", "eval_pointer_random_mix_mixed_results.jsonl"),
    ("Pointer\nK=6, mixed", "eval_pointer_random_mix_k6_mixed_results.jsonl"),
]

fig, ax = plt.subplots(figsize=(6.0, 3.0))

rng_trial_accs = []  # per condition, the 11 non-control trials (cycle + 10 random)
control_accs = []
for label, fname in conditions_fig1:
    rows = load(fname)
    by_trial = {r["trial"]: r["cell_accuracy"] for r in rows}
    control_accs.append(by_trial["control_all_A"])
    others = [v for k, v in by_trial.items() if k != "control_all_A"]
    rng_trial_accs.append(others)

xs = list(range(len(conditions_fig1)))
for i, accs in enumerate(rng_trial_accs):
    jitter = [i + (0.08 * ((j % 5) - 2)) for j in range(len(accs))]
    ax.scatter(jitter, [100 * a for a in accs], color="#4C72B0", alpha=0.75,
               s=22, zorder=3, label="recombined trials" if i == 0 else None)
ax.scatter(xs, [100 * a for a in control_accs], color="#DD8452", marker="D",
           s=40, zorder=4, label="control (single known alphabet)")

ax.set_xticks(xs)
ax.set_xticklabels([c[0] for c in conditions_fig1])
ax.set_ylabel("Cell accuracy (%)")
ax.set_ylim(0, 102)
ax.axhline(100 / 9, color="gray", linestyle=":", linewidth=1, zorder=1)
ax.text(len(xs) - 0.55, 100 / 9 + 2, "chance", fontsize=7.5, color="gray")
ax.legend(loc="lower left", frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_mixed_alphabet.pdf"))
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: zero-shot transfer to held-out alphabet E, across conditions.
# ---------------------------------------------------------------------------
def find_E_row(rows, mode_key="mode", mode_val=None, eval_key="eval_alphabet"):
    for r in rows:
        if r.get(eval_key) == "E" and (mode_val is None or r.get(mode_key) == mode_val):
            return r
    raise ValueError("no matching row found")


conditions_fig2 = []

# Classifier, single alphabet (naive cross, Experiment 0). eval_results.jsonl
# only has alphabets A/B (E is reserved for the K=6 experiments), so this
# condition stands in as "classifier, no multi-alphabet pretraining at all".
rows0 = load("eval_results.jsonl")
r = [x for x in rows0 if x["mode"] == "naive_cross" and x["trained_alphabet"] == "A"][0]
conditions_fig2.append(("Classifier\nK=1\n(cross-alphabet)", r["cell_accuracy"]))

# Classifier, K=6 fixed, zero-shot on E.
rows_ft = load("eval_finetune_results.jsonl")
r = [x for x in rows_ft if x.get("mode") == "zero_shot" and x.get("eval_alphabet") == "E"][0]
conditions_fig2.append(("Classifier\nK=6, fixed", r["cell_accuracy"]))

# Pointer, K=6 fixed.
r = find_E_row(load("eval_pointer_results.jsonl"), mode_val="zero_shot")
conditions_fig2.append(("Pointer\nK=6, fixed", r["cell_accuracy"]))

# Pointer, K=25 random-mix.
r = find_E_row(load("eval_pointer_random_mix_results.jsonl"), mode_val="zero_shot")
conditions_fig2.append(("Pointer\nK=25, mixed", r["cell_accuracy"]))

# Pointer, K=6 random-mix (disentangling follow-up).
r = find_E_row(load("eval_pointer_random_mix_k6_results.jsonl"), mode_val="zero_shot")
conditions_fig2.append(("Pointer\nK=6, mixed", r["cell_accuracy"]))

fig, ax = plt.subplots(figsize=(6.0, 3.0))
xs = list(range(len(conditions_fig2)))
heights = [100 * v for _, v in conditions_fig2]
ax.bar(xs, heights, color="#4C72B0", width=0.55, zorder=3)
ax.axhline(100 / 9, color="gray", linestyle=":", linewidth=1, zorder=1)
ax.text(len(xs) - 0.65, 100 / 9 + 0.6, "chance", fontsize=7.5, color="gray")
ax.set_xticks(xs)
ax.set_xticklabels([c[0] for c in conditions_fig2])
ax.set_ylabel("Cell accuracy on held-out E (%)")
ax.set_ylim(0, max(heights) + 4)
for x, h in zip(xs, heights):
    ax.text(x, h + 0.3, f"{h:.1f}", ha="center", fontsize=7.5)
ax.text(0.5, 0.92, "valid rate = 0% in every condition shown",
        transform=ax.transAxes, ha="center", fontsize=8, color="#333333")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_zeroshot_E.pdf"))
plt.close(fig)

print("wrote", os.path.join(OUT, "fig_mixed_alphabet.pdf"))
print("wrote", os.path.join(OUT, "fig_zeroshot_E.pdf"))
