"""RAG dashboard: single HTML file showing retrieval quality."""
import json


def render(metrics: dict, reliability: list, out_path: str):
    m = metrics
    rel_rows = "".join(
        f"<tr><td>{r['bin']}</td><td>{r['accuracy']}</td><td>{r['n']}</td></tr>"
        for r in reliability)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>VeriRAG Dashboard</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 1em;color:#1a1a1a}}
h1{{border-bottom:3px solid #2563eb;padding-bottom:.3em}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:1em;margin:1em 0}}
.card{{border:1px solid #ddd;border-radius:8px;padding:1em;text-align:center}}
.card .v{{font-size:2em;font-weight:700;color:#2563eb}}
.card .l{{font-size:.85em;color:#555}}
table{{border-collapse:collapse;width:100%;margin:1em 0}}
th,td{{border:1px solid #ddd;padding:.5em;text-align:center}}
th{{background:#f3f4f6}}
.warn{{background:#fef3c7;border:1px solid #f59e0b;border-radius:8px;padding:1em;margin:1em 0}}
</style></head><body>
<h1>VeriRAG — Retrieval Quality Dashboard</h1>
<div class="cards">
<div class="card"><div class="v">{m['hit@10']:.2f}</div><div class="l">Hit@10 (answerable)</div></div>
<div class="card"><div class="v">{m['mrr']:.2f}</div><div class="l">MRR</div></div>
<div class="card"><div class="v">{m['ndcg@10']:.2f}</div><div class="l">NDCG@10</div></div>
<div class="card"><div class="v">{m['refusal_precision']:.2f}</div><div class="l">Refusal precision</div></div>
<div class="card"><div class="v">{m['refusal_recall']:.2f}</div><div class="l">Refusal recall ({m['adversarial_handled']}/{m['n_adversarial']} adversarial)</div></div>
<div class="card"><div class="v">{m['answer_rate']:.2f}</div><div class="l">Answer rate (answerable)</div></div>
<div class="card"><div class="v">{m['mean_grounding']:.2f}</div><div class="l">Mean claim grounding</div></div>
<div class="card"><div class="v">{m['mean_latency_ms']:.0f}ms</div><div class="l">Mean latency</div></div>
<div class="card"><div class="v">{m['n_total']}</div><div class="l">Golden cases</div></div>
</div>
<h2>Calibration reliability</h2>
<p>Predicted P(correct) vs observed accuracy per bin. A calibrated system hugs the diagonal.</p>
<table><tr><th>Bin center</th><th>Observed accuracy</th><th>n</th></tr>{rel_rows}</table>
<div class="warn"><b>Reading this dashboard:</b> Hit@10/MRR/NDCG measure retrieval on answerable
questions. Refusal precision/recall measure honesty on adversarial questions the corpus cannot
answer. A good system is high on both — it answers what it knows and refuses what it doesn't.</div>
</body></html>"""
    with open(out_path, "w") as f:
        f.write(html)
    return out_path
