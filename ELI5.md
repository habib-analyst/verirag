# VeriRAG — Explain Like I'm 5

## The problem, with a story

Imagine you ask a librarian a question. A good librarian does three things:
1. Finds the right books (retrieval),
2. Tells you exactly which page each fact came from (citations),
3. Says **"I don't know"** when the library doesn't have the answer (refusal).

Most AI assistants are like a librarian who *never* says "I don't know." Ask about a book the library doesn't own, and they'll confidently invent an answer — with fake page numbers. That's a hallucination, and in medicine, law, or engineering, it's dangerous.

## What VeriRAG does

VeriRAG is the honest librarian. For every question:

1. **Searches two ways** (hybrid): by meaning (dense vectors) *and* by keywords (BM25). Two nets catch more fish than one.
2. **Re-reads the top candidates carefully** (cross-encoder reranking): like re-reading the most promising pages before answering.
3. **Checks every single claim** (grounding ledger): each sentence in the answer gets a score — "how well does the source actually support this?" — plus the exact document and page.
4. **Decides whether to answer at all** (calibrated refusal): instead of one magic threshold, it combines five different signals into a calibrated probability — "how likely is this answer correct?" If it's too low, it refuses *and tells you why* (no evidence? weak evidence? off-topic? false premise?).
5. **Proves it on a test** (60 golden questions): 40 it should answer, 20 traps it should refuse. The scorecard is public.

## The key insight

Everyone tells their RAG system "refuse if the score is below X." But researchers measured this and found **it doesn't work** — answerable and unanswerable questions score in overlapping ranges, so no single X separates them. VeriRAG's fix: combine *five* signals and *calibrate* the combination, like a doctor who doesn't rely on one symptom but weighs five test results together.

## Try it

```bash
pip install -r requirements.txt
python3 demo.py
```

You'll see it answer a pump question with `[MAN-001, p.1]` citations — and politely refuse a question about a pump model that doesn't exist, explaining *why*.
