# eval_latency.py
#
# Ops eval: how long does each stage take?
#   dense retrieval      - vector store top-k (no LLM, no network)
#   reranked retrieval   - dense fetch + cross-encoder rerank (no LLM, no network)
#   pipeline pass        - ONE real run per query with the active retriever, timed in
#                          three parts: retrieval, generation, end-to-end.
# The reranker's own overhead is (reranked retrieval - dense retrieval).
#
# Why one pass: it makes a single LLM call per query. Your free Groq quota is a
# small daily token budget, and LLM calls are the only thing here that spends it.
#
# Latency should reflect the model's speed, not rate-limit waiting, so Groq retries
# are switched OFF here: a call that hits a rate limit is counted as an error, not
# as a slow call. A short pause between LLM calls (outside the timer) keeps you
# under the per-minute limit. If the DAILY limit is hit, the pass stops and the
# retrieval numbers are still reported.
#
# Run from the project root:
#   python -m evals.eval_latency 5          # 5 spread-out goldens (recommended)
#   python -m evals.eval_latency 5 norerank # skip the reranker stage

import os

os.environ["GROQ_MAX_RETRIES"] = "0"  # must be set before src.config is imported

import json
import statistics
import sys
import time
from datetime import datetime

from evals.harness import ANSWER_SET_NAMES, RESULTS_DIR, load_goldens
from src.config import USE_RERANKER
from src.generator import generate
from src.retriever import get_retriever

P95_BUDGET_SECONDS = 5.0     # end-to-end p95 target; adjust to what your app needs
LLM_PAUSE_SECONDS = 20       # rest between LLM calls (not timed); lower it on a paid Groq tier


def percentile(values, p):
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(times, errors=0):
    if not times:
        return {"n": 0, "errors": errors}
    return {
        "n": len(times),
        "errors": errors,
        "mean_ms": 1000 * statistics.mean(times),
        "p50_ms": 1000 * percentile(times, 50),
        "p95_ms": 1000 * percentile(times, 95),
        "max_ms": 1000 * max(times),
    }


def is_daily_limit(error):
    text = str(error).lower()
    return "per day" in text or "(tpd)" in text


def time_retrieval(retriever, queries):
    """No LLM involved, so there's nothing to rate-limit."""
    retriever.invoke(queries[0])  # warm-up, not timed
    times = []
    for q in queries:
        start = time.perf_counter()
        retriever.invoke(q)
        times.append(time.perf_counter() - start)
    return summarize(times)


def time_pipeline_pass(retriever, queries):
    """One retrieve + generate per query; returns summaries for the three parts and a status note."""
    retrieval, generation, total = [], [], []
    errors, note = 0, ""

    def one_pass(q):
        t0 = time.perf_counter()
        docs = retriever.invoke(q)
        t1 = time.perf_counter()
        generate(q, docs)
        t2 = time.perf_counter()
        return t1 - t0, t2 - t1, t2 - t0

    try:
        one_pass(queries[0])  # warm-up (connection setup, caches), not timed
    except Exception as e:
        if is_daily_limit(e):
            return None, f"stopped: Groq DAILY token limit reached before timing started ({str(e)[:160]})"
        print(f"  warm-up failed: {str(e)[:120]}")

    for q in queries:
        time.sleep(LLM_PAUSE_SECONDS)  # outside the timer
        try:
            r, g, t = one_pass(q)
        except Exception as e:
            if is_daily_limit(e):
                note = f"stopped early: Groq DAILY token limit reached ({str(e)[:160]})"
                print(f"  {note}")
                break
            errors += 1
            print(f"  error on '{q[:40]}': {str(e)[:100]}")
            continue
        retrieval.append(r)
        generation.append(g)
        total.append(t)

    return {
        "pipeline: retrieval": summarize(retrieval, errors),
        "pipeline: generation": summarize(generation, errors),
        "pipeline: end-to-end": summarize(total, errors),
    }, note


def run_local(queries, with_reranker=True):
    summary, dense = {}, get_retriever()

    print("\nTiming: dense retrieval")
    summary["dense retrieval"] = time_retrieval(dense, queries)

    active = dense
    if with_reranker:
        from src.reranker import RerankingRetriever

        reranking = RerankingRetriever()
        print("Timing: reranked retrieval")
        summary["reranked retrieval"] = time_retrieval(reranking, queries)
        if USE_RERANKER:
            active = reranking

    print(f"Timing: pipeline pass ({'reranking' if active is not dense else 'dense'} retriever, "
          f"{LLM_PAUSE_SECONDS}s rest between LLM calls)")
    passes, note = time_pipeline_pass(active, queries)
    if passes:
        summary.update(passes)
    return summary, note


def print_summary(summary, note=""):
    print("\n=== latency summary (milliseconds) ===")
    print(f"{'Stage':<24} {'Mean':>8} {'P50':>8} {'P95':>8} {'Max':>8} {'N':>4}")
    print("-" * 64)
    for name, s in summary.items():
        if not s["n"]:
            print(f"{name:<24} no successful calls ({s['errors']} errors)")
            continue
        flag = f"  ({s['errors']} errors)" if s["errors"] else ""
        print(f"{name:<24} {s['mean_ms']:8.0f} {s['p50_ms']:8.0f} {s['p95_ms']:8.0f} {s['max_ms']:8.0f} {s['n']:>4}{flag}")

    dense, reranked = summary.get("dense retrieval"), summary.get("reranked retrieval")
    if dense and reranked and dense["n"] and reranked["n"]:
        print(f"\nReranker overhead: about {reranked['mean_ms'] - dense['mean_ms']:.0f} ms per query (mean)")

    e2e = summary.get("pipeline: end-to-end")
    if e2e and e2e["n"]:
        verdict = "within" if e2e["p95_ms"] / 1000 <= P95_BUDGET_SECONDS else "OVER"
        print(f"End-to-end p95 {e2e['p95_ms'] / 1000:.1f}s is {verdict} the {P95_BUDGET_SECONDS:.0f}s budget "
              f"(with only {e2e['n']} samples, p95 is close to the max)")
    elif note:
        print("\nLLM stages were not fully measured; see the note below.")
    if note:
        print(f"Note: {note}")


def save(summary, note):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "latency.json"
    payload = {"run_at": datetime.now().isoformat(timespec="seconds"), "p95_budget_s": P95_BUDGET_SECONDS,
               "note": note, "summary": summary}
    path.write_text(json.dumps(payload, indent=2))
    print(f"Saved to {path}")


if __name__ == "__main__":
    goldens = load_goldens(ANSWER_SET_NAMES, "ideal_answer")
    queries = [g["query"] for g in goldens]

    summary, note = run_local(queries, with_reranker="norerank" not in sys.argv[1:])
    print_summary(summary, note)
    save(summary, note)