"""Late-interaction scoring (ColBERT-style MaxSim, lightweight).

Full ColBERT needs a dedicated token encoder; here we reuse the dense
model's token embeddings via the underlying transformer: for each query
token, take the max cosine similarity over document tokens, then mean.
This captures the fine-grained term matching that bi-encoders miss
(e.g. model numbers like "VX-400" matching exactly).

Honest scope: this is a *simplified* late-interaction layer, not a
trained ColBERT. PAPER.md reports its measured contribution.
"""
import numpy as np


class LateInteractionScorer:
    def __init__(self, index):
        self.ix = index
        self._tok = None
        self._tmodel = None

    def _load(self):
        if self._tmodel is not None:
            return
        import os
        os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
        os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
        from transformers import AutoTokenizer, AutoModel
        import torch
        name = "sentence-transformers/all-MiniLM-L6-v2"
        self._tok = AutoTokenizer.from_pretrained(name)
        self._tmodel = AutoModel.from_pretrained(name)
        self._tmodel.eval()

    def _token_emb(self, texts):
        self._load()
        import torch
        out = []
        with torch.no_grad():
            for t in texts:
                enc = self._tok(t, return_tensors="pt", truncation=True,
                                max_length=128)
                h = self._tmodel(**enc).last_hidden_state[0]  # (L, d)
                h = h / (h.norm(dim=1, keepdim=True) + 1e-9)
                out.append(h.numpy())
        return out

    def maxsim(self, query: str, chunks) -> list:
        """Return MaxSim scores aligned with chunks list."""
        q_emb = self._token_emb([query])[0]          # (Lq, d)
        d_embs = self._token_emb([c.text for c in chunks])
        scores = []
        for d in d_embs:                              # (Ld, d)
            sim = q_emb @ d.T                         # (Lq, Ld)
            scores.append(float(sim.max(axis=1).mean()))
        return scores
