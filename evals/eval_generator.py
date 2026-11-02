# eval_generator.py
#
# Does the generator behave, GIVEN the ideal context? The golden ideal_context
# passages go straight into the generator, so retrieval can't be blamed.
#   Faithfulness      - answer says only what the context says
#   Answer relevancy  - answer actually addresses the question
#   Citation accuracy - each cited [p. N] is a label that was in the context
#   Cited rate        - share of answers that cite at least once
#
# The ideal_context passages have no real PDF pages, so they are labelled
# [p. 1], [p. 2], ... in order. Citation accuracy here tests that the model
# follows the citation format and doesn't invent labels; real page citations
# are checked in eval_pipeline.py.
#
# Run from the project root:
#   python -m evals.eval_generator        # all goldens
#   python -m evals.eval_generator 10     # spread-out sample of 10 (cheaper)

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from langchain_core.documents import Document

from evals.harness import (
    CONTEXT_SET_NAMES,
    EVAL_KWARGS,
    JUDGE_MODEL,
    THRESHOLD,
    citation_stats,
    citation_summary,
    find_file,
    get_judge,
    load_goldens,
    print_citation_problems,
    print_summary,
    strip_citations,
    summarize_by_metric,
)
from src.config import LLM_MODEL
from src.generator import generate


CITATION_PROBLEMS = []  # filled by run(), printed after the summary


def run(generate_fn=generate):
    CITATION_PROBLEMS.clear()
    # 1. LOAD the golden set --- queries with the ideal context passages
    goldens = load_goldens(CONTEXT_SET_NAMES, "ideal_context")

    # 2. RUN THE INJECTED GENERATOR on the ideal context for each question.
    #    Page labels are the passage order: metadata page 0 -> [p. 1], and so on.
    test_cases, citation_rows = [], []
    for g in goldens:
        docs = [Document(page_content=text, metadata={"page": i}) for i, text in enumerate(g["ideal_context"])]
        answer = generate_fn(g["query"], docs)

        valid = set(range(1, len(docs) + 1))
        stats = citation_stats(answer, valid_pages=valid)
        citation_rows.append(stats)
        CITATION_PROBLEMS.append((g["id"], answer, stats[0], stats[1], valid))
        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=strip_citations(answer),  # judge the content, not the [p. N] tags
                retrieval_context=g["ideal_context"],
            )
        )

    # 3. THE METRICS --- faithfulness (no made-up claims) and answer relevancy
    judge = get_judge()
    metrics = [
        FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False),
    ]

    # 4. EVALUATE --- every metric on every case, printed report.
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "generator_model": LLM_MODEL,
            "judge_model": JUDGE_MODEL,
            "context": "ideal_context",
            "golden_set": find_file(CONTEXT_SET_NAMES).name,
        },
        **EVAL_KWARGS,
    )

    summary = summarize_by_metric(result)
    summary.update(citation_summary(citation_rows))  # code-based checks, no LLM
    return summary


def run_local():
    """Standalone convenience: use the real generator."""
    return run(generate)


if __name__ == "__main__":
    print_summary("generator", run_local())
    print_citation_problems(CITATION_PROBLEMS)