# eval_retriever_reranker.py
#
# Does reranking improve retrieval QUALITY? Runs the SAME retriever eval twice,
# injecting a different retriever each time, then compares the two summaries:
#   dense     - top-k straight from the vector store
#   reranker  - fetch more candidates, rerank with a cross-encoder, keep top-k
#
# (Latency is measured separately in eval_latency.py.)
# This is an experiment, not a CI gate: run it when you change the reranker
# or want evidence for USE_RERANKER.
#
# Run from the project root:
#   python -m evals.eval_retriever_reranker        # all goldens
#   python -m evals.eval_retriever_reranker 10     # spread-out sample of 10

from evals.eval_retriever import run as run_retriever_eval
from evals.harness import THRESHOLD, print_summary
from src.config import FETCH_K
from src.reranker import RerankingRetriever
from src.retriever import get_retriever


def run_local():
    dense = run_retriever_eval(get_retriever(), "dense")
    reranked = run_retriever_eval(RerankingRetriever(), "reranker")
    return {"dense": dense, "reranker": reranked}


def compare(dense, reranked):
    print(f"\n=== dense vs reranker (reranker fetches {FETCH_K}, keeps the best) ===")
    print(f"{'Metric':<24} {'Dense':>7} {'Reranker':>9} {'Change':>8}")
    print("-" * 51)

    changes = []
    for metric in dense:
        d, r = dense[metric]["mean"], reranked.get(metric, {}).get("mean")
        if d is None or r is None:
            print(f"{metric:<24} {'n/a':>7} {'n/a':>9} {'n/a':>8}")
            continue
        print(f"{metric:<24} {d:7.2f} {r:9.2f} {r - d:+8.2f}")
        changes.append(r - d)

    if not changes:
        print("\nVerdict: no scores to compare; check the judge errors above.")
    elif any(c > 0.02 for c in changes) and not any(c < -0.02 for c in changes):
        print("\nVerdict: reranking improves quality. Weigh that against its latency (evals.eval_latency).")
    elif all(abs(c) <= 0.02 for c in changes):
        print("\nVerdict: no measurable quality difference on this set.")
    else:
        print("\nVerdict: mixed result; read the per-question reasons before deciding.")

    recall = dense.get("Contextual Recall", {}).get("mean")
    if recall is not None and recall < THRESHOLD:
        print("Note: dense recall is low, so also look at chunking and embeddings; reranking only reorders what is retrieved.")
    print("With few questions, differences of a few points are noise.")


if __name__ == "__main__":
    out = run_local()
    print_summary("retriever_dense", out["dense"])
    print_summary("retriever_reranker", out["reranker"])
    compare(out["dense"], out["reranker"])
