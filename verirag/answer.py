"""Answer assembly: extractive, citation-bound, page-precise.

No generative LLM in the $0 path: the answer is composed of the
best-supported sentences from evidence chunks, each carrying a
[DocID, p.N] citation. A generative LLM can be plugged in via
`generator_fn(claims, evidence)` — but every generated claim still
passes through the GroundingLedger before release.

Parent-document retrieval: each cited chunk can be expanded to its
full page for context (parent = page here; pages are the parents
of chunks by construction).
"""
import re


def best_sentences(chunk_text: str, query: str, k: int = 2) -> list:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk_text)
             if len(s.strip().split()) >= 6]
    qtok = set(query.lower().split())
    scored = sorted(
        sents,
        key=lambda s: -len(qtok & set(s.lower().split())),
    )
    return scored[:k]


def assemble_answer(query: str, ranked_chunks, ledger) -> dict:
    """ranked_chunks: list of (chunk, score). Returns answer dict."""
    claims = []
    for chunk, score in ranked_chunks[:3]:
        for sent in best_sentences(chunk.text, query, k=2):
            claims.append((sent, chunk))
    # dedupe near-identical sentences
    seen, uniq = set(), []
    for sent, ch in claims:
        key = sent.lower()[:60]
        if key not in seen:
            seen.add(key)
            uniq.append((sent, ch))
    parts = []
    for sent, ch in uniq[:4]:
        parts.append(f"{sent} [{ch.doc_id}, p.{ch.page_num}]")
    answer = " ".join(parts)
    rows = ledger.ledger(answer, [c for c, _ in ranked_chunks[:5]])
    return {"answer": answer, "citations": [
        {"doc_id": ch.doc_id, "page": ch.page_num, "section": ch.section}
        for _, ch in uniq[:4]], "ledger": rows}


def get_parent_page(chunk, docs) -> str:
    """Parent-document retrieval: full page text for a chunk's page."""
    for d in docs:
        if d.doc_id == chunk.doc_id:
            for p in d.pages:
                if p.page_num == chunk.page_num:
                    return p.text
    return ""
