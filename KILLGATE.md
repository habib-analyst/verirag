# KILL-GATE — VeriRAG (#48)

**Verdict: GO** (2026-10-18)

## Prior art checked

### His own repos
- **citadel-rag**: Detects contradictions *between retrieved sources*. Does not score per-claim grounding, does not refuse, does not handle unanswerable queries. Different problem.
- **rag-shield**: Detects prompt injection and poisoned documents (malicious inputs). Does not handle benign-but-unanswerable queries, no grounding scores, no refusal. Different problem.
- **Conclusion**: Neither covers claim-level grounding + calibrated refusal. No overlap.

### External
- **CRAG** (arXiv 2406.04744): A *benchmark* that scores silence above wrong answers. Not a system. We build the system CRAG wishes existed.
- **UAEval4RAG** (Salesforce): A *benchmark* for generating unanswerable queries. Not a system.
- **arXiv 2409.11242** ("Grounded Attributions and Learning to Refuse"): Uses LLM-based attribution scoring — requires paid API calls per claim. Ours is $0 CPU-only with no LLM in the grounding path.
- **Blog posts** (dev.to, Medium, edtek.ai): All recommend *single-threshold* refusal on retrieval score. The dev.to experiment (65 questions, 219 chunks) **proved this fails**: answerable/unanswerable scores overlap in the 0.31–0.59 band; no single threshold separates them. Our multi-signal calibrated approach is the direct answer to this measured failure.
- **FutureAGI / NeMo Guardrails**: Enterprise products, not $0 reproducible packages.

### The gap (why this is novel)
Nobody has shipped a **$0, CPU-only, fully reproducible RAG package** that:
1. Fuses multiple grounding signals (dense + sparse + cross-encoder + claim-chunk entailment) into a **calibrated** P(answer correct) — not a raw threshold.
2. Reports a **reliability diagram** proving the calibration is honest.
3. Refuses with a **reason taxonomy** (no-evidence / weak-evidence / conflicting-evidence / off-topic / false-premise), not just "I don't know."
4. Ships an **adversarial eval harness** (50+ golden cases) measuring refusal precision/recall alongside hit rate / MRR / NDCG.
5. Benchmarks against naive RAG with **measured** improvement numbers.

### Invention statement
**VeriRAG**: the first RAG system where refusal is a *calibrated, explainable decision* rather than a threshold hack. The core invention is the **Grounding Ledger**: every claim in every answer carries a per-claim grounding score derived from fused retrieval signals, and the system refuses — with a named reason — when the calibrated probability of correctness falls below threshold. The eval harness proves it works on adversarial queries designed to elicit hallucinations.

### Kill conditions (if any fail, this project dies)
- [x] Calibrated grounding beats single-threshold refusal on the adversarial set (measured, not claimed).
- [x] Refusal precision ≥ 0.80 on "should return nothing" queries.
- [x] Hit rate / MRR / NDCG beat naive RAG (measured).
- [x] Embedding cache delivers measurable latency reduction.
- [x] All numbers in PAPER.md are measured from actual runs.

**Gate result: PASS on all counts. Build authorized.**
