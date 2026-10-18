"""Eval harness: hit rate, MRR, NDCG, refusal precision/recall, grounding.

Golden cases: eval/golden.jsonl (40 answerable + 20 adversarial).
Metrics:
  - hit@k: fraction of answerable queries with a gold page in top-k
  - MRR: mean reciprocal rank of first gold page
  - NDCG@k: normalized discounted cumulative gain
  - refusal precision: refused & truly unanswerable / all refused
  - refusal recall: refused & truly unanswerable / all unanswerable
  - answer rate: answered & truly answerable / all answerable
  - grounding rate: mean per-claim grounding score on answered queries
"""
import json
import numpy as np


def _gold_ids(case):
    return set(case.get("pages", []))


def _ranked_ids(result_chunks):
    # result_chunks: list of (chunk, score) from retrieval stage
    return [f"{c.doc_id} p.{c.page_num}" for c, _ in result_chunks]


def hit_at_k(ranked, gold, k):
    return 1.0 if any(g in ranked[:k] for g in gold) else 0.0


def mrr(ranked, gold):
    for i, rid in enumerate(ranked):
        if rid in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked, gold, k):
    dcg = sum(1.0 / np.log2(i + 2) for i, rid in enumerate(ranked[:k])
              if rid in gold)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


class EvalHarness:
    def __init__(self, pipeline):
        self.pipe = pipeline

    def run(self, golden_path, top_k=10, verbose=False):
        cases = [json.loads(l) for l in open(golden_path) if l.strip()]
        rows = []
        for case in cases:
            res = self.pipe.ask(case["query"], top_k=top_k,
                                web_fallback=False)
            # retrieval-stage metrics need raw ranked chunks; re-run retrieval
            cands = self.pipe.retriever.search(case["query"], top_k=50)
            reranked = self.pipe.reranker.rerank(case["query"], cands,
                                                top_k=top_k)
            ranked = _ranked_ids(reranked)
            gold = _gold_ids(case)
            row = {
                "id": case["id"],
                "answerable": case["answerable"],
                "refused": res["refused"],
                "hit@1": hit_at_k(ranked, gold, 1),
                "hit@5": hit_at_k(ranked, gold, 5),
                "hit@10": hit_at_k(ranked, gold, 10),
                "mrr": mrr(ranked, gold),
                "ndcg@10": ndcg_at_k(ranked, gold, 10),
                "p_correct": res.get("p_correct"),
                "reason": res.get("refusal_reason"),
                "latency_ms": res.get("latency_ms"),
            }
            if not res["refused"]:
                lg = res.get("ledger", [])
                row["grounding"] = float(np.mean(
                    [r["grounding_score"] for r in lg])) if lg else 0.0
            rows.append(row)
            if verbose:
                print(case["id"], "refused" if res["refused"] else "answered",
                      f"hit@10={row['hit@10']:.0f}")
        return self.summarize(rows)

    @staticmethod
    def summarize(rows):
        ans = [r for r in rows if r["answerable"]]
        adv = [r for r in rows if not r["answerable"]]
        refused_adv = [r for r in adv if r["refused"]]
        refused_all = [r for r in rows if r["refused"]]
        answered_ans = [r for r in ans if not r["refused"]]

        def mean(rs, k):
            return float(np.mean([r[k] for r in rs])) if rs else 0.0

        return {
            "n_total": len(rows),
            "n_answerable": len(ans),
            "n_adversarial": len(adv),
            "hit@1": mean(ans, "hit@1"),
            "hit@5": mean(ans, "hit@5"),
            "hit@10": mean(ans, "hit@10"),
            "mrr": mean(ans, "mrr"),
            "ndcg@10": mean(ans, "ndcg@10"),
            "refusal_precision": (len(refused_adv) / len(refused_all)
                                  if refused_all else 0.0),
            "refusal_recall": (len(refused_adv) / len(adv) if adv else 0.0),
            "answer_rate": (len(answered_ans) / len(ans) if ans else 0.0),
            "mean_grounding": mean(answered_ans, "grounding"),
            "mean_latency_ms": mean(rows, "latency_ms"),
            "adversarial_handled": len(refused_adv),
        }
