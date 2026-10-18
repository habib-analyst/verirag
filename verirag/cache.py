"""Embedding cache: query embeddings memoized in memory + on disk.

Measured latency reduction is reported by eval (target: 80%).
"""
import hashlib
import json
import os
import time
import numpy as np


class EmbeddingCache:
    def __init__(self, cache_dir=None):
        self.mem = {}
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.t_hit = 0.0
        self.t_miss = 0.0

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def _disk_path(self, key):
        return os.path.join(self.cache_dir, key + ".npy") if self.cache_dir else None

    def get(self, text: str, encode_fn):
        """Return embedding, using cache when possible. encode_fn(text)->np array."""
        key = self._key(text)
        t0 = time.perf_counter()
        if key in self.mem:
            self.hits += 1
            self.t_hit += time.perf_counter() - t0
            return self.mem[key]
        dp = self._disk_path(key)
        if dp and os.path.exists(dp):
            v = np.load(dp)
            self.mem[key] = v
            self.hits += 1
            self.t_hit += time.perf_counter() - t0
            return v
        v = encode_fn(text)
        self.mem[key] = v
        if dp:
            np.save(dp, v)
        self.misses += 1
        self.t_miss += time.perf_counter() - t0
        return v

    def stats(self):
        total = self.hits + self.misses
        avg_hit = self.t_hit / max(1, self.hits)
        avg_miss = self.t_miss / max(1, self.misses)
        reduction = (1 - avg_hit / max(avg_miss, 1e-9)) if self.misses else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / max(1, total),
            "avg_hit_ms": avg_hit * 1000,
            "avg_miss_ms": avg_miss * 1000,
            "latency_reduction": reduction,
        }
