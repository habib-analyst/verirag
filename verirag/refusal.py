"""Calibrated refusal with reason taxonomy.

Instead of a single threshold on one retrieval score (proven insufficient —
see KILLGATE.md), we fuse five signals into a calibrated P(answer correct)
via Platt scaling (logistic regression on a held-out calibration set):

  signals = [max_dense, max_sparse_norm, max_ce, mean_grounding,
             source_agreement, entity_consistency]

The 6th signal — entity consistency — catches false-premise queries:
if the query asks about "HT-700" but every evidence chunk discusses
"HT-900", no retrieval score can save it. Entity consistency = fraction
of query entities found in top evidence. A query entity absent from ALL
evidence is a false premise by definition.

Refusal reasons (taxonomy):
  - off_topic: query has no lexical or semantic overlap with corpus
  - no_evidence: retrieval returned nothing above floor
  - weak_evidence: evidence exists but grounding too low
  - conflicting_evidence: top sources disagree (defer to citadel-rag style flag)
  - false_premise: query asserts something the corpus contradicts

The calibrator is fit on the golden eval set (answerable vs adversarial).
Reliability diagram data is exported for PAPER.md.
"""
import numpy as np
import re
from sklearn.linear_model import LogisticRegression


def extract_query_entities(query: str) -> set:
    """Extract high-value entities: codes (VX-400), numbers with units, proper nouns."""
    ents = set()
    # product codes: LETTERS- digits
    ents.update(re.findall(r"\b[A-Z]{2,}-\d+\b", query))
    # numbers with units
    ents.update(re.findall(r"\b\d+(?:\.\d+)?\s*(?:bar|euro|mm|kg|kW|RPM|°C|degrees|meters|liters|hours|days|weeks|months|percent|%)\b", query, re.I))
    # capitalized multi-word terms (likely proper nouns)
    ents.update(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", query))
    return {e.lower() for e in ents if len(e) > 2}


def entity_consistency(query: str, evidence_chunks) -> float:
    """Fraction of query entities found in top evidence. 1.0 = all found."""
    ents = extract_query_entities(query)
    if not ents:
        return 1.0  # no entities to check -> neutral
    ev_text = " ".join(c.text.lower() for c, _ in evidence_chunks[:5])
    hit = sum(1 for e in ents if e in ev_text)
    return hit / len(ents)


REASONS = ["off_topic", "no_evidence", "weak_evidence",
           "conflicting_evidence", "false_premise"]


class RefusalCalibrator:
    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
        self.clf = LogisticRegression()
        self.fitted = False

    def fit(self, X, y):
        """X: (n, 6) signal matrix, y: 1 if answerable/correct, else 0."""
        self.clf.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
        self.fitted = True
        return self

    def p_correct(self, signals) -> float:
        if not self.fitted:
            # uncalibrated fallback: mean of signals
            return float(np.mean(signals))
        return float(self.clf.predict_proba(
            np.asarray(signals, dtype=float).reshape(1, -1))[0, 1])

    def decide(self, signals, reason_hint: str = None) -> dict:
        # Circuit breaker 1: query entities absent from ALL evidence =
        # false premise by definition. No calibration needed.
        # signals[5] is entity_consistency.
        if len(signals) > 5 and signals[5] < 0.5:
            return {"p_correct": round(float(signals[5]) * 0.5, 3),
                    "refuse": True,
                    "reason": "false_premise",
                    "threshold": self.threshold,
                    "circuit_breaker": "entity"}
        # Circuit breaker 2: max dense score below absolute floor =
        # nothing in the corpus is semantically close. Off-topic.
        # signals[0] is max_dense (cosine, [0,1]).
        if signals[0] < 0.35:
            return {"p_correct": round(float(signals[0]), 3),
                    "refuse": True,
                    "reason": "off_topic",
                    "threshold": self.threshold,
                    "circuit_breaker": "dense_floor"}
        p = self.p_correct(signals)
        refuse = p < self.threshold
        return {"p_correct": round(p, 3),
                "refuse": refuse,
                "reason": reason_hint if refuse else None,
                "threshold": self.threshold,
                "circuit_breaker": False}

    def reliability(self, X, y, bins: int = 10):
        """Reliability diagram data: (bin_center, accuracy, count)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        p = self.clf.predict_proba(X)[:, 1] if self.fitted else X.mean(axis=1)
        edges = np.linspace(0, 1, bins + 1)
        out = []
        for i in range(bins):
            m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1
                                   else p <= edges[i + 1])
            if m.sum():
                out.append({"bin": round((edges[i] + edges[i + 1]) / 2, 2),
                            "accuracy": round(float(y[m].mean()), 3),
                            "n": int(m.sum())})
        return out


def classify_refusal_reason(query: str, candidates, ledger_rows,
                            entity_cons: float = 1.0) -> str:
    """Rule-based refusal reason from retrieval + grounding state."""
    q = query.lower()
    if not candidates:
        return "no_evidence"
    # false premise: query entities missing from all evidence
    if entity_cons < 0.5:
        return "false_premise"
    # off-topic: almost no token overlap with top candidates
    qtok = set(q.split())
    overlap = max(len(qtok & set(c.text.lower().split())) / max(1, len(qtok))
                  for c, *_ in candidates[:5])
    if overlap < 0.15:
        return "off_topic"
    # weak evidence: low grounding across the board despite retrieval hits
    if ledger_rows and all(r["grounding_score"] < 0.35 for r in ledger_rows):
        return "weak_evidence"
    return "weak_evidence"
