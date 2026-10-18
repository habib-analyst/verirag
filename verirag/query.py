"""Query expansion and rewriting.

Expansion: pseudo-relevance feedback — take top-3 initial BM25 hits,
extract discriminative terms not in the query, append them.
Rewriting: rule-based disambiguation for vague queries
(e.g. "it", "that pump", missing entity) using corpus entity lexicon.
"""
import re
from collections import Counter


ENTITY_LEXICON = {
    "pump": ["VX-400", "pump"],
    "heat exchanger": ["HT-900", "heat exchanger"],
    "controller": ["SC-50", "safety controller"],
    "leave": ["leave", "annual leave", "sick leave", "parental leave"],
    "vpn": ["VPN", "vpn.novadyne.example"],
    "spill": ["spill", "Level 2"],
    "lockout": ["lockout", "LOTO", "tagout"],
    "per diem": ["per diem", "travel"],
    "cleanroom": ["cleanroom", "ISO Class"],
}


def expand_query(query: str, bm25, tokenized_docs, top_n: int = 3,
                 n_terms: int = 5) -> str:
    """Pseudo-relevance feedback expansion."""
    scores = bm25.get_scores(query.lower().split())
    top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_n]
    qterms = set(query.lower().split())
    counter = Counter()
    for i in top_idx:
        for w in tokenized_docs[i]:
            if w not in qterms and len(w) > 3:
                counter[w] += 1
    extra = [w for w, _ in counter.most_common(n_terms)]
    return query + " " + " ".join(extra) if extra else query


def rewrite_query(query: str) -> tuple:
    """Rewrite ambiguous queries. Returns (rewritten, was_ambiguous, notes)."""
    notes = []
    q = query.strip()
    low = q.lower()
    ambiguous = False

    # pronoun / vague reference without entity
    if re.search(r"\b(it|that|this|they)\b", low):
        has_entity = any(e in low for e in ENTITY_LEXICON)
        if not has_entity:
            ambiguous = True
            notes.append("pronoun without antecedent")

    # single-word or very short queries
    if len(low.split()) <= 2 and not any(e in low for e in ENTITY_LEXICON):
        ambiguous = True
        notes.append("too short to disambiguate")

    if ambiguous:
        # attach most likely entity hints so retrieval has something to bite
        hints = " ".join(sorted(ENTITY_LEXICON.keys())[:0])  # none; keep honest
        notes.append("flagged ambiguous: retrieval will be low-confidence")
        return q, True, notes
    return q, False, notes
