# eval_pipeline.py
#
# End-to-end RAG triad on the real pipeline (retrieve -> generate):
#   Contextual relevancy - query  <-> retrieved chunks  (is retrieval on topic?)
#   Faithfulness         - chunks <-> answer            (groundedness: no made-up claims?)
#   Answer relevancy     - query  <-> answer            (does it answer what was asked?)
# plus a code-based check that every cited [p. N] is a page that was retrieved.
#
# The weakest link points to the stage to fix:
#   low contextual relevancy -> retriever (chunking, embeddings, k, reranker)
#   low faithfulness         -> generator (prompt, model)
#   low answer relevancy     -> generator/prompt, or poor retrieval upstream
#
# Run from the project root:
#   python -m evals.eval_pipeline         # all goldens
#   python -m evals.eval_pipeline 10      # spread-out sample of 10 (cheaper)

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from evals.harness import (
    ANSWER_SET_NAMES,
    EVAL_KWARGS,
    JUDGE_MODEL,
    THRESHOLD,
    citation_stats,
    citation_summary,
    find_file,
    get_judge,
    load_goldens,
    per_case_means,
    print_citation_problems,
    print_summary,
    strip_citations,
    summarize_by_metric,
)
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, LLM_MODEL, TOP_K, USE_RERANKER
from src.pipeline import ask

EXTRA_ROWS = {"Citation Accuracy", "Cited Rate", "RAG Triad (mean)"}


CITATION_PROBLEMS = []  # filled by run(), printed after the summary


def run(ask_fn=ask):
    CITATION_PROBLEMS.clear()
    # 1. LOAD the golden set --- the queries (ideal_answer is kept for the correctness eval later)
    goldens = load_goldens(ANSWER_SET_NAMES, "ideal_answer")

    # 2. RUN THE INJECTED PIPELINE on each question; ask_fn returns {"answer", "pages", "docs"}.
    test_cases, citation_rows = [], []
    for g in goldens:
        out = ask_fn(g["query"])

        valid = set(out["pages"])
        stats = citation_stats(out["answer"], valid_pages=valid)
        citation_rows.append(stats)
        CITATION_PROBLEMS.append((g["id"], out["answer"], stats[0], stats[1], valid))
        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=strip_citations(out["answer"]),
                expected_output=g["ideal_answer"],
                retrieval_context=[d.page_content for d in out["docs"]],
            )
        )

    # 3. THE METRICS --- the three sides of the RAG triad
    judge = get_judge()
    metrics = [
        ContextualRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False),
        FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False),
    ]

    # 4. EVALUATE --- every metric on every case, printed report.
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": "reranker" if USE_RERANKER else "dense",
            "embedding_model": EMBED_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": TOP_K,
            "generator_model": LLM_MODEL,
            "judge_model": JUDGE_MODEL,
            "golden_set": find_file(ANSWER_SET_NAMES).name,
        },
        **EVAL_KWARGS,
    )

    summary = summarize_by_metric(result)

    # combined triad score per question (mean of its three metric scores)
    triad = [m for m in per_case_means(result) if m is not None]
    if triad:
        summary["RAG Triad (mean)"] = {
            "mean": sum(triad) / len(triad),
            "pass_rate": sum(1 for m in triad if m >= THRESHOLD) / len(triad),
            "n": len(triad),
            "errors": 0,
        }

    summary.update(citation_summary(citation_rows))  # code-based check, no LLM
    return summary


def run_local():
    """Standalone convenience: use the real pipeline."""
    return run(ask)


if __name__ == "__main__":
    summary = run_local()
    print_summary("pipeline", summary)
    print_citation_problems(CITATION_PROBLEMS)

    links = {k: v["mean"] for k, v in summary.items() if k not in EXTRA_ROWS and v["mean"] is not None}
    if links:
        weakest = min(links, key=links.get)
        print(f"\nWeakest link: {weakest} ({links[weakest]:.2f})")