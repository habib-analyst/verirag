"""Per-claim grounding: every claim in an answer gets a score.

A "claim" is an atomic factual statement extracted from the answer.
Grounding score for a claim = max over evidence chunks of a fused
support signal:
  support = 0.5 * cross_encoder(claim, chunk)
          + 0.3 * token_overlap(claim, chunk)
          + 0.2 * entity_match(claim, chunk)
normalized to [0,1].

This is the Grounding Ledger: claim -> (score, best chunk, page).
"""
import re
import numpy as np


def extract_claims(answer: str) -> list:
    """Split answer into atomic claims (sentence-level)."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    # drop pure citation markers / refusals
    return [s for s in sents if len(s.split()) >= 4]


def _token_overlap(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)


def _entity_match(claim: str, chunk_text: str) -> float:
    # numbers, codes, and capitalized terms are high-value entities
    ents = set(re.findall(r"[A-Z]{2,}(?:-\d+)?|\d+(?:\.\d+)?\s?(?:bar|euro|mm|kg|kW|RPM|°C|degrees|meters|liters|hours|days|weeks|months|percent|%)?",
                          claim))
    if not ents:
        return 0.5  # neutral when no entities to check
    hit = sum(1 for e in ents if e.lower() in chunk_text.lower())
    return hit / len(ents)


class GroundingLedger:
    def __init__(self, reranker=None):
        self.reranker = reranker  # optional cross-encoder for support

    def score_claim(self, claim: str, chunks) -> tuple:
        """Return (best_score, best_chunk)."""
        best, best_c = 0.0, None
        for c in chunks:
            if self.reranker is not None and self.reranker._ce is not None:
                s_ce = float(self.reranker._ce.predict(
                    [(claim, c.text)], show_progress_bar=False)[0])
                s_ce = 1 / (1 + np.exp(-s_ce))  # sigmoid to [0,1]
            else:
                s_ce = _token_overlap(claim, c.text)
            s = (0.5 * s_ce
                 + 0.3 * _token_overlap(claim, c.text)
                 + 0.2 * _entity_match(claim, c.text))
            if s > best:
                best, best_c = s, c
        return best, best_c

    def ledger(self, answer: str, evidence_chunks, threshold: float = 0.45):
        """Build the grounding ledger for an answer."""
        rows = []
        for claim in extract_claims(answer):
            score, chunk = self.score_claim(claim, evidence_chunks)
            rows.append({
                "claim": claim,
                "grounding_score": round(score, 3),
                "grounded": score >= threshold,
                "source": (f"{chunk.doc_id} p.{chunk.page_num}"
                           if chunk else None),
            })
        return rows
