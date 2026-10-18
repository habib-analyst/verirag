"""Hybrid retrieval: dense + sparse score fusion, metadata filtering.

Fusion: min-max normalize each signal to [0,1], weighted sum.
Default weights: dense 0.6, sparse 0.4 (measured best on golden set).
"""
import numpy as np


def _minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class HybridRetriever:
    def __init__(self, index, cache=None, w_dense=0.6, w_sparse=0.4):
        self.ix = index
        self.cache = cache
        self.w_dense = w_dense
        self.w_sparse = w_sparse

    def _encode_query(self, query: str):
        def enc(t):
            v = self.ix.model.encode([t], show_progress_bar=False,
                                     convert_to_numpy=True)[0]
            return v / (np.linalg.norm(v) + 1e-9)
        if self.cache:
            return self.cache.get(query, lambda t: enc(t))
        return enc(query)

    def search(self, query: str, top_k: int = 50, filters: dict = None):
        """Return list of (chunk, fused_score, dense_score, sparse_score)."""
        qv = self._encode_query(query)
        dense_scores = self.ix._dense @ qv
        sparse_scores = np.array(self.ix._bm25.get_scores(query.lower().split()))

        fused = self.w_dense * _minmax(dense_scores) + \
                self.w_sparse * _minmax(sparse_scores)

        order = np.argsort(-fused)
        out = []
        for i in order:
            c = self.ix.chunks[i]
            if filters and not self._passes(c, filters):
                continue
            out.append((c, float(fused[i]), float(dense_scores[i]),
                        float(sparse_scores[i])))
            if len(out) >= top_k:
                break
        return out

    @staticmethod
    def _passes(chunk, filters: dict) -> bool:
        for k, v in filters.items():
            attr = {"doc_type": chunk.doc_type, "department": chunk.department,
                    "doc_id": chunk.doc_id}.get(k)
            if attr is None:
                continue
            if isinstance(v, (list, tuple, set)):
                if attr not in v:
                    return False
            elif attr != v:
                return False
        return True
