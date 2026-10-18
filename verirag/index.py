"""Hybrid index: dense (sentence-transformers) + sparse (BM25).

Embeddings are cached on disk (see cache.py) so rebuilds are instant.
"""
import json
import os
import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import chunk_pages
from .corpus import build_corpus, get_all_pages


class HybridIndex:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 cache_dir=None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.chunks = []
        self._dense = None       # np array (n_chunks, dim)
        self._bm25 = None
        self._tokenized = []
        self._model = None

    @property
    def model(self):
        if self._model is None:
            # sanitize proxy env (known VM issue: bracketed IPv6 in no_proxy)
            os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
            os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build(self, docs=None):
        docs = docs or build_corpus()
        pages = get_all_pages(docs)
        doc_meta = {d.doc_id: d for d in docs}
        chunks = chunk_pages(pages)
        for c in chunks:
            m = doc_meta[c.doc_id]
            c.doc_title = m.title
            c.doc_type = m.doc_type
            c.department = m.department
            c.date = m.date
        self.chunks = chunks

        # sparse
        self._tokenized = [c.text.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized)

        # dense (with disk cache)
        texts = [c.text for c in chunks]
        self._dense = self._cached_encode(texts)
        return self

    def _cache_path(self):
        if not self.cache_dir:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, "dense_emb.npy")

    def _cached_encode(self, texts):
        p = self._cache_path()
        if p and os.path.exists(p):
            arr = np.load(p)
            if arr.shape[0] == len(texts):
                return arr
        emb = self.model.encode(texts, show_progress_bar=False,
                                convert_to_numpy=True)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        if p:
            np.save(p, emb)
        return emb

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "chunks.json"), "w") as f:
            json.dump([c.__dict__ for c in self.chunks], f)
        np.save(os.path.join(path, "dense.npy"), self._dense)

    @classmethod
    def load(cls, path, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        ix = cls(model_name=model_name)
        with open(os.path.join(path, "chunks.json")) as f:
            from .chunking import Chunk
            ix.chunks = [Chunk(**d) for d in json.load(f)]
        ix._dense = np.load(os.path.join(path, "dense.npy"))
        ix._tokenized = [c.text.lower().split() for c in ix.chunks]
        ix._bm25 = BM25Okapi(ix._tokenized)
        return ix
