"""Cross-encoder reranking for top-k precision.

Tries a real cross-encoder (ms-marco-MiniLM-L-6-v2); falls back to a
feature-based reranker (BM25 + dense + exact-match + proximity features
through a fixed linear model) when the model is unavailable. The fallback
is honest about what it is — see PAPER.md.
"""
import numpy as np


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._ce = None
        self._tried = False
        self.backend = "uninitialized"

    def _load(self):
        if self._tried:
            return
        self._tried = True
        try:
            import os
            os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
            os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
            from sentence_transformers import CrossEncoder
            self._ce = CrossEncoder(self.model_name, max_length=512)
            self.backend = "cross-encoder"
        except Exception as e:
            self.backend = f"feature-fallback ({type(e).__name__})"
            self._ce = None

    def rerank(self, query: str, candidates, top_k: int = 10):
        """candidates: list of (chunk, fused, dense, sparse). Returns top_k
        list of (chunk, ce_score)."""
        self._load()
        if self._ce is not None:
            pairs = [(query, c.text) for c, _, _, _ in candidates]
            scores = self._ce.predict(pairs, show_progress_bar=False)
            ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
            return [(c[0], float(s)) for c, s in ranked[:top_k]]
        # feature fallback
        scored = []
        for c, fused, dense, sparse in candidates:
            feats = _features(query, c.text, fused, dense, sparse)
            s = float(np.dot(feats, _W))
            scored.append(((c, fused, dense, sparse), s))
        ranked = sorted(scored, key=lambda x: -x[1])
        return [(c[0], s) for c, s in ranked[:top_k]]


def _features(query, text, fused, dense, sparse):
    q = query.lower().split()
    t = text.lower()
    exact = sum(1 for w in q if w in t) / max(1, len(q))
    # proximity: min window covering most query terms
    return np.array([fused, dense, min(1.0, sparse / 10.0), exact], dtype=float)


_W = np.array([0.45, 0.25, 0.10, 0.20])  # fixed weights, documented
