# VeriRAG — RAG with Calibrated Grounding and Principled Refusal

**The 30-second why:** Most RAG systems hallucinate when the corpus can't answer — they guess instead of refusing. VeriRAG is the first $0, CPU-only RAG package where *refusal is a calibrated, explainable decision*: every claim carries a grounding score with page-precise citations, and the system refuses with a named reason when evidence runs out. Measured on 60 golden cases (40 answerable + 20 adversarial traps).

## What it does

- **Hybrid search** — dense (sentence-transformers) + sparse (BM25) with learned fusion weights
- **Cross-encoder reranking** — top-50 → top-10 for precision
- **Late interaction** — ColBERT-style token MaxSim for fine-grained matching (model numbers, codes)
- **Page-level citations** — every claim cited as `[DOC-ID, p.N]`; chunks never cross page boundaries
- **Per-claim grounding ledger** — each answer sentence scored against its best evidence
- **Calibrated refusal** — 5 signals fused via Platt scaling into P(correct); refuses with reason taxonomy (no-evidence / weak-evidence / conflicting-evidence / off-topic / false-premise)
- **Query expansion + rewriting** — pseudo-relevance feedback; ambiguous queries flagged
- **Metadata filtering** — scope retrieval by doc type, department, doc ID
- **Parent-document retrieval** — expand any chunk to its full page
- **Embedding cache** — measured latency reduction (see PAPER.md)
- **Eval harness** — 60 golden cases, hit@k / MRR / NDCG / refusal precision-recall
- **Dashboard** — single-file HTML with reliability diagram

## Quickstart

```bash
pip install -r requirements.txt
python3 demo.py          # 30-second demo: answer + principled refusal
python3 -m pytest tests/ -q
python3 run_eval.py       # full eval vs naive RAG (~5 min on CPU)
```

## How refusal works

```
query → hybrid retrieve (50) → rerank (10) → provisional answer
      → grounding ledger per claim → 5-signal calibration → P(correct)
      → P ≥ 0.5: answer with citations
      → P < 0.5: refuse with reason + web-search fallback note
```

## Measured results (see PAPER.md)

| Metric | Naive RAG | VeriRAG |
|---|---|---|
| Hit@10 | 0.975 | **1.000** |
| MRR | 0.879 | **0.942** |
| NDCG@10 | 0.904 | **0.957** |
| Adversarial hallucination rate | 100% (20/20) | **50%** (10/20 refused) |
| Refusal precision / recall | n/a | **0.91 / 0.50** |
| Answer rate (answerable) | 100% | 97.5% (39/40) |
| Mean claim grounding | — | 0.85 |
| Embedding cache latency reduction | — | 99.8% |

*(Numbers from `run_eval.py` → `eval/results.json`, 60 golden cases.)*

## Novelty

Single-threshold refusal is proven insufficient (answerable/unanswerable scores overlap 0.31–0.59; see KILLGATE.md). VeriRAG fuses five signals into a *calibrated* decision and explains *why* it refused. Different from citadel-rag (source contradictions) and rag-shield (malicious inputs): this handles benign-but-unanswerable queries.

## License

MIT
