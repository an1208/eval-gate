"""Fast, offline tests (no API keys, no models) - run before the expensive evals."""

from evals import gate
from evals.harness import citation_stats, citation_summary, rate_limit_wait, strip_citations


def test_strip_citations():
    assert strip_citations("Sales were X [p. 1] and Y [p. 2, 7].").split() == ["Sales", "were", "X", "and", "Y", "."]


def test_citation_stats_counts_valid_pages():
    acc, n = citation_stats("X [p. 1] and Y [p. 2, 7]", {1, 2, 3})
    assert n == 3
    assert abs(acc - 2 / 3) < 1e-9


def test_citation_stats_no_citations():
    assert citation_stats("No citations here.", {1, 2}) == (0.0, 0)


def test_citation_summary():
    s = citation_summary([(1.0, 2), (0.0, 0)])
    assert s["Citation Accuracy"]["mean"] == 0.5
    assert s["Cited Rate"]["mean"] == 0.5


def _results(score):
    return {"retriever": {"summary": {"Contextual Recall": {"mean": score}}}}


THRESHOLDS = {"retriever": {"Contextual Recall": 0.7}}


def test_gate_passes_when_above_minimum_and_baseline():
    rows = gate.check(THRESHOLDS, _results(0.85), {"retriever": {"Contextual Recall": 0.86}}, 0.05)
    assert rows[0]["status"] == "PASS"


def test_gate_fails_below_minimum():
    rows = gate.check(THRESHOLDS, _results(0.60), {}, 0.05)
    assert rows[0]["status"] == "FAIL" and "below minimum" in rows[0]["note"]


def test_gate_fails_on_regression_even_above_minimum():
    rows = gate.check(THRESHOLDS, _results(0.75), {"retriever": {"Contextual Recall": 0.90}}, 0.05)
    assert rows[0]["status"] == "FAIL" and "regressed" in rows[0]["note"]


def test_gate_fails_when_suite_missing():
    rows = gate.check(THRESHOLDS, {"retriever": None}, {}, 0.05)
    assert rows[0]["status"] == "FAIL"


def test_rate_limit_wait_parses_groq_message():
    msg = "Error code: 429 - Rate limit reached ... Please try again in 8.0175s. Need more tokens?"
    assert abs(rate_limit_wait(Exception(msg), 0) - 9.0175) < 1e-6


def test_rate_limit_wait_minutes_and_ms():
    assert abs(rate_limit_wait(Exception("429 ... try again in 1m2.5s"), 0) - 63.5) < 1e-6
    assert abs(rate_limit_wait(Exception("429 ... try again in 250ms"), 0) - 1.25) < 1e-6


def test_rate_limit_wait_ignores_other_errors():
    assert rate_limit_wait(Exception("401 invalid api key"), 0) is None
    assert rate_limit_wait(Exception("429 too many requests"), 3) == 8


def test_gate_fails_when_too_many_judge_errors():
    results = {"retriever": {"summary": {"Contextual Recall": {"mean": 0.95, "n": 10, "errors": 4}}}}
    rows = gate.check(THRESHOLDS, results, {}, 0.05, 0.2)
    assert rows[0]["status"] == "FAIL" and "judge errors" in rows[0]["note"]


def test_gate_tolerates_a_few_judge_errors():
    results = {"retriever": {"summary": {"Contextual Recall": {"mean": 0.95, "n": 10, "errors": 1}}}}
    assert gate.check(THRESHOLDS, results, {}, 0.05, 0.2)[0]["status"] == "PASS"


def test_normalize_citations_rewrites_variants():
    from src.generator import normalize_citations

    assert normalize_citations("Sales were $391,035 million\u3010p. 32\u3011.") == "Sales were $391,035 million[p. 32]."
    assert normalize_citations("X [p.32] and Y\u3010p. 12,14\u3011") == "X [p. 32] and Y[p. 12, 14]"
    assert normalize_citations("Already fine [p. 5].") == "Already fine [p. 5]."
    assert normalize_citations("No citations here (p is a variable).") == "No citations here (p is a variable)."


def test_normalize_citations_handles_repeated_p_and_spaces():
    from src.generator import normalize_citations

    assert normalize_citations("X were $391,035 million [ p. 32, p. 26 ].") == "X were $391,035 million [p. 32, 26]."
    assert normalize_citations("See [pages 3-4] and [pp. 7, 9]") == "See [p. 3-4] and [p. 7, 9]"
    assert normalize_citations("array[p] and [pricing] and [p. note 5]") == "array[p] and [pricing] and [p. note 5]"