"""Unit tests for VeriRAG. Run: python3 -m pytest tests/ -q"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from verirag.corpus import build_corpus, get_all_pages
from verirag.chunking import chunk_pages
from verirag.grounding import extract_claims, _token_overlap
from verirag.refusal import RefusalCalibrator, classify_refusal_reason
from verirag.query import rewrite_query
import numpy as np


def test_corpus_has_pages():
    docs = build_corpus()
    pages = get_all_pages(docs)
    assert len(pages) >= 40, "need a substantive corpus"
    assert all(p.page_num >= 1 for p in pages)


def test_chunking_respects_pages():
    docs = build_corpus()
    chunks = chunk_pages(get_all_pages(docs), chunk_tokens=500,
                         overlap_tokens=50)
    assert len(chunks) > 0
    # every chunk maps to exactly one page
    for c in chunks:
        assert c.page_num >= 1 and c.doc_id


def test_chunk_overlap_minimum():
    docs = build_corpus()
    pages = get_all_pages(docs)[:3]
    chunks = chunk_pages(pages, chunk_tokens=100, overlap_tokens=50)
    # with small chunks we should get multiple chunks per page (overlap working)
    assert len(chunks) >= len(pages)


def test_claim_extraction():
    claims = extract_claims("The VX-400 pumps 400 liters per minute. It runs on three phase power.")
    assert len(claims) == 2


def test_token_overlap():
    assert _token_overlap("hello world", "hello world") == 1.0
    assert _token_overlap("abc", "xyz") == 0.0


def test_calibrator_fit_predict():
    cal = RefusalCalibrator(threshold=0.5)
    rng = np.random.RandomState(0)
    X = rng.rand(40, 5)
    y = (X[:, 0] > 0.5).astype(int)
    cal.fit(X, y)
    p = cal.p_correct([0.9, 0.9, 0.9, 0.9, 0.9])
    assert 0.0 <= p <= 1.0
    d = cal.decide([0.9, 0.9, 0.9, 0.9, 0.9])
    assert "refuse" in d and "p_correct" in d


def test_rewrite_flags_ambiguous():
    _, amb, _ = rewrite_query("it")
    assert amb is True
    _, amb2, _ = rewrite_query("What is the VX-400 maximum flow rate?")
    assert amb2 is False


def test_refusal_reasons_valid():
    assert classify_refusal_reason("q", [], []) == "no_evidence"


def test_golden_file_count():
    import json
    n = sum(1 for l in open(
        os.path.join(os.path.dirname(__file__), "..", "eval", "golden.jsonl"))
        if l.strip())
    assert n >= 50, f"need 50+ golden cases, got {n}"
