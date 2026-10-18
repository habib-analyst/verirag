"""VeriRAG demo: 30-second why.

Run: python3 demo.py
Shows: an answerable question (with page citations) and an
adversarial question (principled refusal with reason).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verirag.corpus import build_corpus
from verirag.index import HybridIndex
from verirag.pipeline import VeriRAG
from verirag.refusal import RefusalCalibrator
from run_eval import fit_calibrator, GOLDEN, CACHE_DIR


def main():
    print("Building VeriRAG index...")
    docs = build_corpus()
    ix = HybridIndex(cache_dir=CACHE_DIR).build(docs)
    pipe = VeriRAG(ix, docs=docs)
    print("Fitting refusal calibrator...")
    fit_calibrator(pipe, GOLDEN)

    print("\n" + "=" * 60)
    print("Q: What is the maximum flow rate of the VX-400 pump?")
    r = pipe.ask("What is the maximum flow rate of the VX-400 pump?")
    print("A:", r["answer"])
    print("Citations:", r["citations"])
    print(f"Grounded claims: {r['grounded_claims']}/{r['total_claims']}, "
          f"P(correct)={r['p_correct']}")

    print("\n" + "=" * 60)
    print("Q: What is the maximum flow rate of the VX-900 pump?")
    r = pipe.ask("What is the maximum flow rate of the VX-900 pump?")
    print("Refused:", r["refused"])
    print("Reason:", r["refusal_reason"])
    print("P(correct):", r["p_correct"])
    print("Note:", r.get("fallback_note", ""))
    print("=" * 60)


if __name__ == "__main__":
    main()
