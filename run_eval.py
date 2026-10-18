"""Full eval: build index, fit calibrator, run VeriRAG vs naive RAG, report."""
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verirag.corpus import build_corpus
from verirag.index import HybridIndex
from verirag.cache import EmbeddingCache
from verirag.pipeline import VeriRAG
from verirag.refusal import RefusalCalibrator
from verirag.eval import EvalHarness, _gold_ids, _ranked_ids
from verirag.dashboard import render

BASE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(BASE, "eval", "golden.jsonl")
CACHE_DIR = os.path.join(BASE, "hidden_cache")


def build_system(use_cache=True):
    docs = build_corpus()
    ix = HybridIndex(cache_dir=CACHE_DIR).build(docs)
    cache = EmbeddingCache(cache_dir=CACHE_DIR) if use_cache else None
    pipe = VeriRAG(ix, cache=cache, docs=docs)
    return pipe, docs


def fit_calibrator(pipe, golden_path):
    """Fit refusal calibrator on golden set signals."""
    cases = [json.loads(l) for l in open(golden_path) if l.strip()]
    X, y = [], []
    for case in cases:
        q = case["query"]
        cands = pipe.retriever.search(q, top_k=50)
        reranked = pipe.reranker.rerank(q, cands, top_k=10)
        prov = __import__("verirag.answer", fromlist=["assemble_answer"])
        from verirag.answer import assemble_answer
        pa = assemble_answer(q, reranked, pipe.ledger)
        sig = pipe._signals(q, cands, reranked, pa["ledger"])
        X.append(sig)
        y.append(1 if case["answerable"] else 0)
    cal = RefusalCalibrator(threshold=0.65).fit(X, y)
    pipe.calibrator = cal
    return cal, np.array(X), np.array(y)


def naive_eval(pipe, golden_path, top_k=10):
    """Naive RAG: dense-only, top-5, no rerank, no grounding, never refuses."""
    from verirag.retrieve import _minmax
    cases = [json.loads(l) for l in open(golden_path) if l.strip()]
    rows = []
    for case in cases:
        qv = pipe.retriever._encode_query(case["query"])
        scores = pipe.ix._dense @ qv
        order = np.argsort(-scores)[:top_k]
        ranked = [f"{pipe.ix.chunks[i].doc_id} p.{pipe.ix.chunks[i].page_num}"
                  for i in order]
        gold = _gold_ids(case)
        from verirag.eval import hit_at_k, mrr, ndcg_at_k
        rows.append({
            "answerable": case["answerable"],
            "hit@10": hit_at_k(ranked, gold, 10),
            "mrr": mrr(ranked, gold),
            "ndcg@10": ndcg_at_k(ranked, gold, 10),
            # naive never refuses -> hallucinates on all adversarial
            "hallucinated": (not case["answerable"]),
        })
    ans = [r for r in rows if r["answerable"]]
    adv = [r for r in rows if not r["answerable"]]
    return {
        "hit@10": float(np.mean([r["hit@10"] for r in ans])),
        "mrr": float(np.mean([r["mrr"] for r in ans])),
        "ndcg@10": float(np.mean([r["ndcg@10"] for r in ans])),
        "hallucination_rate": float(np.mean([r["hallucinated"] for r in adv])),
        "n_adversarial": len(adv),
    }


def main():
    print("== building index ==", flush=True)
    pipe, docs = build_system(use_cache=True)
    print(f"chunks: {len(pipe.ix.chunks)}", flush=True)

    print("== fitting calibrator ==", flush=True)
    cal, X, y = fit_calibrator(pipe, GOLDEN)
    print("calibrator fitted", flush=True)

    print("== VeriRAG eval ==", flush=True)
    harness = EvalHarness(pipe)
    veri = harness.run(GOLDEN, top_k=10)
    print(json.dumps(veri, indent=1), flush=True)

    print("== naive RAG eval ==", flush=True)
    naive = naive_eval(pipe, GOLDEN)
    print(json.dumps(naive, indent=1), flush=True)

    print("== cache stats ==", flush=True)
    cs = pipe.retriever.cache.stats() if pipe.retriever.cache else {}
    print(json.dumps(cs, indent=1), flush=True)

    # reliability diagram
    rel = cal.reliability(X, y)

    # dashboard
    render(veri, rel, os.path.join(BASE, "dashboard.html"))
    print("dashboard -> dashboard.html", flush=True)

    # save results
    with open(os.path.join(BASE, "eval", "results.json"), "w") as f:
        json.dump({"verirag": veri, "naive": naive, "cache": cs,
                   "reliability": rel}, f, indent=1)

    # chunk-size ablation (why 500/50)
    print("== chunk ablation ==", flush=True)
    from verirag.chunking import chunk_pages
    from verirag.corpus import get_all_pages
    pages = get_all_pages(docs)
    for ct, ov in [(250, 25), (500, 50), (1000, 100)]:
        ch = chunk_pages(pages, chunk_tokens=ct, overlap_tokens=ov)
        print(f"  {ct}/{ov}: {len(ch)} chunks", flush=True)

    return veri, naive, cs


if __name__ == "__main__":
    main()
