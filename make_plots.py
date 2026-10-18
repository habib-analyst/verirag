"""Generate publication visuals from eval/results.json."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(BASE, "eval", "results.json")))
v, n = res["verirag"], res["naive"]
os.makedirs(os.path.join(BASE, "assets"), exist_ok=True)


# 1. Retrieval metrics: VeriRAG vs naive
fig, ax = plt.subplots(figsize=(8, 4.5))
xs = ["Hit@10", "MRR", "NDCG@10"]
vv = [v["hit@10"], v["mrr"], v["ndcg@10"]]
nv = [n["hit@10"], n["mrr"], n["ndcg@10"]]
x = np.arange(len(xs))
ax.bar(x - 0.2, vv, 0.4, label="VeriRAG", color="#2563eb")
ax.bar(x + 0.2, nv, 0.4, label="Naive RAG", color="#9ca3af")
ax.set_xticks(x)
ax.set_xticklabels(xs)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Retrieval quality: VeriRAG vs naive RAG (40 answerable queries)")
ax.legend()
for i, (a, b) in enumerate(zip(vv, nv)):
    ax.text(i - 0.2, a + 0.02, f"{a:.2f}", ha="center", fontsize=9)
    ax.text(i + 0.2, b + 0.02, f"{b:.2f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(BASE, "assets", "retrieval.png"), dpi=150)

# 2. Adversarial handling
fig, ax = plt.subplots(figsize=(7, 4.5))
cats = ["Refused\n(correct)", "Answered\n(hallucinated)"]
veri_vals = [v["adversarial_handled"],
             v["n_adversarial"] - v["adversarial_handled"]]
naive_vals = [0, n["n_adversarial"]]
x = np.arange(len(cats))
ax.bar(x - 0.2, veri_vals, 0.4, label="VeriRAG", color="#2563eb")
ax.bar(x + 0.2, naive_vals, 0.4, label="Naive RAG", color="#ef4444")
ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.set_ylabel("Queries (of 20 adversarial)")
ax.set_title("Adversarial queries: who hallucinates?")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(BASE, "assets", "adversarial.png"), dpi=150)

# 3. Reliability diagram
rel = res.get("reliability", [])
if rel:
    fig, ax = plt.subplots(figsize=(6, 6))
    bins = [r["bin"] for r in rel]
    acc = [r["accuracy"] for r in rel]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(bins, acc, "o-", color="#2563eb", label="VeriRAG calibrator")
    ax.set_xlabel("Predicted P(correct)")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Calibration reliability diagram")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "assets", "reliability.png"), dpi=150)

# 4. Latency: cache hit vs miss
cs = res.get("cache", {})
if cs:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Cache hit", "Cache miss"],
           [cs["avg_hit_ms"], cs["avg_miss_ms"]],
           color=["#22c55e", "#f59e0b"])
    ax.set_ylabel("ms per query embedding")
    ax.set_title(f"Embedding cache: {cs['latency_reduction']:.1%} latency reduction")
    for i, val in enumerate([cs["avg_hit_ms"], cs["avg_miss_ms"]]):
        ax.text(i, val + max(1, val * 0.05), f"{val:.2f}ms",
                ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "assets", "cache.png"), dpi=150)

print("plots -> assets/")
