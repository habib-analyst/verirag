"""Chunking: 500 tokens with 50-token overlap.

WHY THIS STRATEGY WORKS (documented, measured in eval/):
1. 500 tokens ≈ one self-contained idea. Smaller chunks (100-200) fragment
   procedures and specs across boundaries, destroying the context a
   cross-encoder needs to judge relevance. Larger chunks (1000+) dilute
   the signal: the answer-relevant sentence drowns in surrounding text
   and BM25 term statistics skew toward long documents.
2. 50-token overlap (≈10%) is the minimum that prevents boundary splits
   from orphaning key facts. A fact spanning a chunk boundary appears
   whole in at least one chunk. Below 50, we measured orphaned facts;
   above 100, index bloat with no recall gain.
3. Page-aligned: chunks never cross page boundaries, so every chunk maps
   to exactly one (document, page) — citations stay precise.

Measured: chunk-size ablation in eval shows 500/50 beats 250/25 and
1000/100 on hit@10 for this corpus (see PAPER.md).
"""
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    page_num: int
    section: str
    text: str
    token_count: int
    doc_type: str
    department: str
    date: str


def _approx_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return max(1, len(text) // 4)


def chunk_pages(pages, chunk_tokens: int = 500, overlap_tokens: int = 50) -> list:
    """Split pages into chunks. Never crosses page boundaries."""
    chunks = []
    for page in pages:
        words = page.text.split()
        # Convert token budget to word budget (~0.75 words/token)
        chunk_words = int(chunk_tokens * 0.75)
        overlap_words = int(overlap_tokens * 0.75)
        step = max(1, chunk_words - overlap_words)
        idx = 0
        n = 0
        for start in range(0, len(words), step):
            piece = words[start:start + chunk_words]
            if not piece:
                break
            text = " ".join(piece)
            # skip tiny trailing fragments (merge into previous by dropping)
            if len(piece) < overlap_words and start > 0:
                break
            chunks.append(Chunk(
                chunk_id=f"{page.doc_id}-p{page.page_num}-c{n}",
                doc_id=page.doc_id,
                doc_title="",  # filled by build_index
                page_num=page.page_num,
                section=page.section,
                text=text,
                token_count=_approx_tokens(text),
                doc_type="",
                department="",
                date="",
            ))
            n += 1
            idx += 1
            if start + chunk_words >= len(words):
                break
    return chunks
