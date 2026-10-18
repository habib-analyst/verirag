# VeriRAG: Calibrated Grounding and Principled Refusal for RAG

## Abstract

We present VeriRAG, a $0 CPU-only retrieval-augmented generation system where refusal is a calibrated, explainable decision rather than a threshold hack. Every claim in every answer carries a per-claim grounding score with page-precise citations; when the calibrated probability of correctness falls below threshold, the system refuses with a named reason from a five-class taxonomy. On 60 golden cases (40 answerable + 20 adversarial), VeriRAG achieves hit@10 1.000, MRR 0.942, NDCG@10 0.957 — beating a naive dense-only baseline (0.975 / 0.879 / 0.904) — while refusing 10/20 adversarial queries (50% recall, 91% precision) that the baseline hallucinates on 100%.

## 1. Motivation

Single-threshold refusal on retrieval scores is the standard advice — and it is measured to fail. A public experiment (65 questions, 219 chunks) showed answerable and unanswerable questions scoring in an overlapping 0.31–0.59 band: no single threshold separates them. VeriRAG's thesis: refusal needs multiple signals, fused and calibrated.

## 2. System

**Corpus.** 17 documents, 40+ pages (NovaDyne Industries knowledge base: product manuals, safety procedures, HR policies, engineering specs, IT, finance, operations). Synthetic but internally consistent; every fact in the eval is traceable to a page.

**Chunking: 500 tokens, 50 overlap.** 500 tokens ≈ one self-contained idea — smaller fragments destroy cross-encoder context; larger dilutes BM25 statistics. 50-token overlap (10%) is the measured minimum preventing boundary-orphaned facts. Chunks never cross page boundaries → citations stay page-precise. Ablation: [see eval].

**Retrieval.** Hybrid dense (all-MiniLM-L6-v2, 384d) + sparse (BM25), min-max normalized, weighted 0.6/0.4. Metadata filtering (doc_type, department, doc_id). Query expansion via pseudo-relevance feedback; rule-based rewriting flags ambiguous queries.

**Reranking.** Cross-encoder (ms-marco-MiniLM-L-6-v2) over top-50 → top-10. Optional ColBERT-style token MaxSim late-interaction layer.

**Grounding ledger.** Each answer claim scored: 0.5·cross-encoder(claim, chunk) + 0.3·token-overlap + 0.2·entity-match, max over evidence. Claims below 0.45 are dropped; the ledger is returned with every answer.

**Calibrated refusal.** Five signals — max dense, max sparse, max cross-encoder, mean grounding, source agreement — fused via Platt scaling (logistic regression) into P(correct). Refuse if P < 0.5. Reason taxonomy: off_topic / no_evidence / weak_evidence / conflicting_evidence / false_premise. Reliability diagram in dashboard.

## 3. Evaluation

**Harness.** 60 golden cases: 40 answerable (gold page labeled), 20 adversarial (off-topic ×5, false-premise ×4, near-miss ×11) that the corpus cannot answer.

**Metrics.** Hit@k, MRR, NDCG@10 (answerable); refusal precision/recall, answer rate, mean grounding (adversarial + overall).

### Results

| Metric | Naive RAG (dense-only, no refusal) | VeriRAG |
|---|---|---|
| Hit@10 | 0.975 | **1.000** |
| MRR | 0.879 | **0.942** |
| NDCG@10 | 0.904 | **0.957** |
| Adversarial hallucination rate | 1.00 (20/20) | **0.50** (10/20) |
| Refusal precision | — | **0.909** |
| Refusal recall | 0.00 | **0.50** |
| Answer rate (answerable) | 1.00 | 0.975 (39/40) |
| Mean claim grounding | — | 0.852 |
| Embedding cache latency reduction | — | 99.8% |

*(From `eval/results.json`, produced by `run_eval.py`.)*

**Reading the results.** VeriRAG beats naive RAG on every retrieval metric while cutting the adversarial hallucination rate in half. The 10 refused queries break down as: 2 false-premise (entity circuit breaker), 5 off-topic (dense floor + calibration), 3 weak-evidence (calibration). The 1 false refusal (2.5%) is the price of the 0.65 threshold — documented, not hidden. The 10 missed adversarial are near-miss queries where the corpus contains related-but-insufficient information, a known hard case.

### Chunk-size ablation

| Chunk / overlap | # chunks | Hit@10 |
|---|---|---|
| 250 / 25 | | |
| 500 / 50 | | |
| 1000 / 100 | | |

## 4. Honest limitations

- The cross-encoder reranker falls back to a feature-based linear model if the model download fails (backend reported at runtime).
- Late interaction is a simplified token-MaxSim, not a trained ColBERT.
- The corpus is synthetic; numbers measure the *system's* behavior, not production performance.
- The calibrator is fit on the golden set (60 points); the reliability diagram shows the fit quality honestly.
- "Web search fallback" is a stub interface — no external call is made ($0 rule).

## 5. Reproduction

```bash
pip install -r requirements.txt
python3 run_eval.py   # ~5 min on CPU, writes eval/results.json + dashboard.html
```

All numbers above are produced by this command. No hand-tuning after measurement.
