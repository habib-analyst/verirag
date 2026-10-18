"""VeriRAG pipeline: query -> answer or principled refusal."""
import time
import numpy as np

from .retrieve import HybridRetriever
from .rerank import CrossEncoderReranker
from .query import expand_query, rewrite_query
from .grounding import GroundingLedger
from .refusal import RefusalCalibrator, classify_refusal_reason
from .answer import assemble_answer


class VeriRAG:
    def __init__(self, index, cache=None, calibrator=None,
                 docs=None, use_late_interaction=False):
        self.ix = index
        self.docs = docs or []
        self.retriever = HybridRetriever(index, cache=cache)
        self.reranker = CrossEncoderReranker()
        self.ledger = GroundingLedger()
        self.calibrator = calibrator or RefusalCalibrator()
        self.use_li = use_late_interaction
        self._li = None

    def _signals(self, query, cands, reranked, ledger_rows):
        from .refusal import entity_consistency
        max_dense = max((d for _, _, d, _ in cands), default=0.0)
        max_sparse = max((s for _, _, _, s in cands), default=0.0)
        max_ce = max((s for _, s in reranked), default=0.0)
        mean_ground = float(np.mean([r["grounding_score"]
                                     for r in ledger_rows])) if ledger_rows else 0.0
        # source agreement: share of top-5 chunks from the majority doc
        top_docs = [c.doc_id for c, _ in reranked[:5]]
        agree = (max(top_docs.count(d) for d in set(top_docs)) / max(1, len(top_docs))
                 if top_docs else 0.0)
        ent_cons = entity_consistency(query, reranked)
        # normalize sparse roughly to [0,1]
        return [max(0.0, min(1.0, max_dense)),
                max(0.0, min(1.0, max_sparse / 10.0)),
                max(0.0, min(1.0, (max_ce + 5) / 10.0)),
                mean_ground, agree, ent_cons]

    def ask(self, query: str, filters: dict = None, top_k: int = 10,
            web_fallback: bool = True) -> dict:
        t0 = time.perf_counter()
        rewritten, ambiguous, notes = rewrite_query(query)
        expanded = expand_query(rewritten, self.ix._bm25,
                                self.ix._tokenized)

        cands = self.retriever.search(expanded, top_k=50, filters=filters)
        reranked = self.reranker.rerank(expanded, cands, top_k=top_k)

        if self.use_li:
            from .late_interaction import LateInteractionScorer
            if self._li is None:
                self._li = LateInteractionScorer(self.ix)
            chunks = [c for c, _ in reranked]
            li_scores = self._li.maxsim(expanded, chunks)
            reranked = sorted(zip(chunks, li_scores), key=lambda x: -x[1])[:top_k]

        # provisional answer for grounding measurement
        prov = assemble_answer(expanded, reranked, self.ledger)
        signals = self._signals(expanded, cands, reranked, prov["ledger"])
        reason = classify_refusal_reason(expanded, cands, prov["ledger"],
                                         entity_cons=signals[5])
        decision = self.calibrator.decide(signals, reason_hint=reason)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if decision["refuse"]:
            out = {"answer": None, "refused": True,
                   "refusal_reason": decision["reason"],
                   "p_correct": decision["p_correct"],
                   "fallback": "web_search" if web_fallback else None,
                   "latency_ms": round(elapsed_ms, 1),
                   "ambiguous_query": ambiguous, "query_notes": notes}
            if web_fallback:
                out["fallback_note"] = (
                    "Confidence below threshold; deferring to web search. "
                    "(Stub: plug a search API here. No external call made.)")
            return out

        result = assemble_answer(expanded, reranked, self.ledger)
        # final grounding gate: drop ungrounded claims
        kept = [r for r in result["ledger"] if r["grounded"]]
        result.update({"refused": False, "p_correct": decision["p_correct"],
                       "latency_ms": round(elapsed_ms, 1),
                       "grounded_claims": len(kept),
                       "total_claims": len(result["ledger"]),
                       "ambiguous_query": ambiguous, "query_notes": notes})
        return result
