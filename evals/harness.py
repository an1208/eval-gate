"""
harness.py - shared plumbing for the eval files

Run every eval from the project root as a module:
    python -m evals.eval_pipeline 10

Golden sets (same 38 queries in both), looked for in goldens/, the project
root and data/:
    golden_retrieve.json  -> id, query, ideal_answer
    golden_generate.json  -> id, query, ideal_context (list of passages)

Pass a number to any eval to run a spread-out sample, e.g. 10 goldens.
"""

import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from src.config import JUDGE_MODEL, JUDGE_REASONING_EFFORT, ROOT

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
THRESHOLD = 0.7  # per-question pass mark for every DeepEval metric

# File names are tried in this order, so either spelling works.
ANSWER_SET_NAMES = ["golden_retrieve.json", "golden_set_retrieval.json", "golden_set_retrieve.json"]
CONTEXT_SET_NAMES = ["golden_generate.json", "golden_set_generate.json"]


# ---------------------------------------------------------------------
# Golden sets
# ---------------------------------------------------------------------
def find_file(names):
    for folder in (ROOT / "goldens", ROOT, ROOT / "data"):
        for name in names:
            if (folder / name).is_file():
                return folder / name
    raise SystemExit(
        f"Couldn't find any of {names} in goldens/, the project root or data/.\n"
        "Put your golden files in goldens/, or edit the names at the top of evals/harness.py."
    )


def load_goldens(names, required_field):
    """Load a golden set; with a number on the command line, take an evenly spread sample."""
    path = find_file(names)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data or required_field not in data[0]:
        raise SystemExit(
            f"{path.name} has no '{required_field}' field. If your two golden files are swapped, "
            "swap the names in ANSWER_SET_NAMES / CONTEXT_SET_NAMES in evals/harness.py."
        )

    limit = next((int(a) for a in sys.argv[1:] if a.isdigit()), None)
    if limit and limit < len(data):
        step = len(data) / limit  # spread, so the qualitative questions at the end are included
        data = [data[int(i * step)] for i in range(limit)]

    print(f"Loaded {len(data)} goldens from {path.relative_to(ROOT)}")
    return data


# ---------------------------------------------------------------------
# Judge: Groq model wrapped for DeepEval
# ---------------------------------------------------------------------
MAX_RETRIES = 8  # attempts per judge call when Groq says "rate limit reached"


def rate_limit_wait(error, attempt):
    """Seconds to wait after a Groq rate-limit (429) error, or None if it isn't one.

    Groq's message says e.g. "Please try again in 8.0175s" (or "1m2.5s", "250ms"),
    so we wait exactly that long (+1s); otherwise back off 1, 2, 4, ... up to 30s.
    """
    text = str(error)
    if "429" not in text and "rate limit" not in text.lower():
        return None
    m = re.search(r"try again in (?:(\d+)m(?!s))?([\d.]+)(ms|s)", text)
    if m:
        minutes, value, unit = m.groups()
        seconds = float(value) / (1000 if unit == "ms" else 1) + 60 * int(minutes or 0)
        return seconds + 1
    return min(2 ** attempt, 30)


def is_truncated(message):
    """True if the model stopped because it hit its length limit (so JSON would be cut off)."""
    return (getattr(message, "response_metadata", None) or {}).get("finish_reason") == "length"


def get_judge():
    from deepeval.models import DeepEvalBaseLLM
    from langchain_groq import ChatGroq

    class GroqJudge(DeepEvalBaseLLM):
        def __init__(self):
            super().__init__(JUDGE_MODEL)  # calls load_model() and stores it in self.model

        def load_model(self):
            kwargs = {}
            if "gpt-oss" in JUDGE_MODEL and JUDGE_REASONING_EFFORT:
                kwargs["reasoning_effort"] = JUDGE_REASONING_EFFORT
            return ChatGroq(model=JUDGE_MODEL, temperature=0, **kwargs)

        def generate(self, prompt: str) -> str:
            cut_off = 0
            for attempt in range(MAX_RETRIES):
                try:
                    msg = self.model.invoke(prompt)
                except Exception as e:
                    wait = rate_limit_wait(e, attempt)
                    if wait is None or attempt == MAX_RETRIES - 1:
                        raise
                    print(f"  [judge] rate limited, waiting {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(wait)
                    continue
                if is_truncated(msg) and cut_off < 2:  # reply was cut off mid-JSON: ask again
                    cut_off += 1
                    print(f"  [judge] reply was cut off, retrying ({cut_off}/2)")
                    continue
                return msg.content
            return msg.content

        async def a_generate(self, prompt: str) -> str:
            cut_off = 0
            for attempt in range(MAX_RETRIES):
                try:
                    msg = await self.model.ainvoke(prompt)
                except Exception as e:
                    wait = rate_limit_wait(e, attempt)
                    if wait is None or attempt == MAX_RETRIES - 1:
                        raise
                    print(f"  [judge] rate limited, waiting {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                    await asyncio.sleep(wait)
                    continue
                if is_truncated(msg) and cut_off < 2:
                    cut_off += 1
                    print(f"  [judge] reply was cut off, retrying ({cut_off}/2)")
                    continue
                return msg.content
            return msg.content

        def get_model_name(self):
            return f"Groq {JUDGE_MODEL}"

    return GroqJudge()


# Run metrics one at a time instead of in parallel, so Groq's rate limits
# aren't hit. Remove this to let DeepEval parallelise (faster, but may 429).
try:
    from deepeval.evaluate.configs import AsyncConfig, CacheConfig, ErrorConfig

    EVAL_KWARGS = {
        "async_config": AsyncConfig(run_async=False),
        # Don't read/write DeepEval's on-disk test-run cache. On Windows its file
        # lock can fail and crash evaluate() with "'NoneType' has no attribute
        # 'test_cases_lookup_map'"; we never reuse cached results anyway.
        "cache_config": CacheConfig(write_cache=False, use_cache=False),
        # If the judge still returns unusable output for one metric, record an error
        # for that metric and carry on, instead of crashing the whole run. The error
        # count shows in the summary, and the CI gate fails if too many occur.
        "error_config": ErrorConfig(ignore_errors=True),
    }
except ImportError:
    EVAL_KWARGS = {}


# ---------------------------------------------------------------------
# Citations - answers cite pages like [p. 12] or [p. 12, 14]
# ---------------------------------------------------------------------
CITATION_RE = re.compile(r"\[p\.?\s*[\d,\s\-–]+\]", re.IGNORECASE)


def strip_citations(answer: str) -> str:
    """Remove [p. N] tags so the judge scores the content, not the tags."""
    return CITATION_RE.sub("", answer).strip()


def citation_stats(answer: str, valid_pages: set):
    """(share of cited pages that were really in the context, number of citations)."""
    cited = []
    for match in CITATION_RE.findall(answer):
        cited += [int(n) for n in re.findall(r"\d+", match)]
    if not cited:
        return 0.0, 0
    return sum(p in valid_pages for p in cited) / len(cited), len(cited)


def citation_summary(stats):
    """Turn [(accuracy, n_cited), ...] into two summary rows."""
    n = len(stats)
    return {
        "Citation Accuracy": {
            "mean": sum(acc for acc, _ in stats) / n,
            "pass_rate": sum(1 for acc, c in stats if c and acc == 1.0) / n,
            "n": n,
            "errors": 0,
        },
        "Cited Rate": {
            "mean": sum(1 for _, c in stats if c) / n,
            "pass_rate": sum(1 for _, c in stats if c) / n,
            "n": n,
            "errors": 0,
        },
    }


# ---------------------------------------------------------------------
# Reading DeepEval results
# ---------------------------------------------------------------------
def summarize_by_metric(result):
    """{metric name: {"mean", "pass_rate", "n", "errors"}} from an evaluate() result."""
    buckets = {}
    for test in result.test_results:
        for m in test.metrics_data or []:
            b = buckets.setdefault(m.name, {"scores": [], "passed": 0, "n": 0, "errors": 0})
            b["n"] += 1
            b["passed"] += 1 if m.success else 0
            b["errors"] += 1 if getattr(m, "error", None) else 0
            if m.score is not None:
                b["scores"].append(m.score)

    return {
        name: {
            "mean": sum(b["scores"]) / len(b["scores"]) if b["scores"] else None,
            "pass_rate": b["passed"] / b["n"],
            "n": b["n"],
            "errors": b["errors"],
        }
        for name, b in buckets.items()
    }


def per_case_means(result):
    """One average score per test case (used for the combined RAG-triad score)."""
    means = []
    for test in result.test_results:
        scores = [m.score for m in (test.metrics_data or []) if m.score is not None]
        means.append(sum(scores) / len(scores) if scores else None)
    return means


def print_summary(name, summary, save=True):
    """Print the summary table and (by default) save it to evals/results/<name>.json."""
    print(f"\n=== {name}: summary (per-question pass mark {THRESHOLD}) ===")
    print(f"{'Metric':<26} {'Mean':>6} {'Pass rate':>10} {'N':>4}")
    print("-" * 49)
    for metric, s in summary.items():
        mean = "  n/a" if s["mean"] is None else f"{s['mean']:5.2f}"
        flag = f"   ({s['errors']} judge errors)" if s.get("errors") else ""
        print(f"{metric:<26} {mean:>6} {s['pass_rate']:>10.0%} {s['n']:>4}{flag}")

    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        slug = re.sub(r"\W+", "_", name).strip("_").lower()
        path = RESULTS_DIR / f"{slug}.json"
        payload = {"run_at": datetime.now().isoformat(timespec="seconds"), "threshold": THRESHOLD, "summary": summary}
        path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved to {path.relative_to(ROOT)}")