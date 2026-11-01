# eval_retriever.py
#
# Does the retriever find what's needed to answer?
#   Contextual recall    - did we miss information needed for the ideal answer?
#   Contextual precision - are the useful chunks ranked above the noise?
#
# Evaluates the ACTIVE retriever (reranking if USE_RERANKER is true).
#
# Run from the project root:
#   python -m evals.eval_retriever        # all goldens
#   python -m evals.eval_retriever 10     # spread-out sample of 10 (cheaper)

from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

from evals.harness import (
    ANSWER_SET_NAMES,
    EVAL_KWARGS,
    JUDGE_MODEL,
    THRESHOLD,
    find_file,
    get_judge,
    load_goldens,
    print_summary,
    summarize_by_metric,
)
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, FETCH_K, TOP_K, USE_RERANKER
from src.pipeline import build_retriever


def run(retriever, retriever_name="dense"):
    # 1. LOAD the golden set --- the fixed, human-authored truth
    goldens = load_goldens(ANSWER_SET_NAMES, "ideal_answer")

    # 2. RUN THE INJECTED RETRIEVER on each question to fill retrieval_context,
    #    then build one test case per golden.
    test_cases = []
    for g in goldens:
        retrieved = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
    judge = get_judge()
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False),
    ]

    # 4. EVALUATE --- every metric on every case, printed report.
    #    hyperparameters travel with the run so the report is tagged with the config.
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": retriever_name,      # "dense" vs "reranker" when you swap it in
            "embedding_model": EMBED_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": TOP_K,
            "fetch_k": FETCH_K if retriever_name == "reranker" else TOP_K,
            "judge_model": JUDGE_MODEL,
            "golden_set": find_file(ANSWER_SET_NAMES).name,
        },
        **EVAL_KWARGS,
    )
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the active retriever, then run."""
    return run(build_retriever(), "reranker" if USE_RERANKER else "dense")


if __name__ == "__main__":
    print_summary("retriever", run_local())
