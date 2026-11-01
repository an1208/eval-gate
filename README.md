# EvalGate: Production-Grade RAG with Reranking and CI-Gated LLM Evaluation

A question-answering system over Apple's FY2024 annual report (Form 10-K), built so that **every change to the retriever, prompt or model has to pass an automated quality gate before it can merge**.

- **RAG pipeline:** PyPDF, recursive chunking, MiniLM embeddings, ChromaDB, optional cross-encoder reranking, Groq LLM with page citations and an explicit "I don't know" behaviour.
- **Layered evals (DeepEval, Groq as judge):** retriever, generator and the end-to-end RAG triad are scored separately, so a drop points at the stage that caused it.
- **CI gate:** a GitHub Actions workflow runs the evals on every pull request and fails the build if a score drops below its minimum or regresses against a committed baseline.

## Architecture

```
PDF -> chunks -> embeddings -> Chroma
                                 |
 question -> dense top-10 (FETCH_K) -> cross-encoder rerank -> top-4 (TOP_K)
                                                                   |
                         prompt (cite pages, refuse if unsure) -> Groq LLM -> answer + pages

 evals/ : golden sets -> retriever eval | generator eval | pipeline eval  -> results/*.json
                                                                              |
                                      thresholds.json + baseline.json -> gate.py -> pass / fail CI
```

## Project layout

```
src/
  config.py        all tunables (chunking, k, models, USE_RERANKER); loads .env
  retriever.py     load -> split -> embed -> Chroma; dense retriever
  reranker.py      RerankingRetriever (over-retrieve, cross-encode, keep top-k)
  generator.py     prompt + LLM; generate(question, docs)
  pipeline.py      the full chain; ask(question) -> answer, pages, docs
evals/
  harness.py                  golden loading, Groq judge, summaries
  eval_retriever.py           contextual recall + precision
  eval_retriever_reranker.py  dense vs reranker quality (experiment, not gated)
  eval_generator.py           faithfulness, answer relevancy, citation accuracy (ideal context in)
  eval_pipeline.py            RAG triad + real-page citation accuracy
  eval_latency.py             per-stage latency (mean / p50 / p95), separate add-on
  gate.py                     the CI gate (absolute minimums + regression vs baseline)
  thresholds.json             minimum score per metric, regression tolerance
  baseline.json               accepted scores (created by --update-baseline; commit it)
goldens/                      golden_retrieve.json, golden_generate.json (you provide)
data/                         apple_2024_report.pdf (you provide)
tests/                        fast offline tests
.github/workflows/eval.yml    the CI workflow
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then put your GROQ_API_KEY in it
```

1. Put the PDF at `data/apple_2024_report.pdf` (or set `PDF_PATH` in `.env`).
2. Put your golden files in `goldens/`: `golden_retrieve.json` (`id`, `query`, `ideal_answer`) and `golden_generate.json` (`id`, `query`, `ideal_context`).

```bash
python -m src.pipeline "What were Apple's total net sales in fiscal 2024?"
python -m src.pipeline                      # interactive
python -m src.retriever --reindex           # rebuild the index after changing chunking/embeddings
```

Run every command from the project root. Delete `chroma_db/` (or use `--reindex`) whenever you change `CHUNK_SIZE`, `CHUNK_OVERLAP` or `EMBED_MODEL`.

## Running the evals

Each LLM-judged metric costs several Groq calls. Add a number to run a spread-out sample of that many goldens.

```bash
python -m evals.eval_retriever 10
python -m evals.eval_generator 10
python -m evals.eval_pipeline 10
python -m evals.eval_retriever_reranker 10   # is reranking worth keeping?
python -m evals.eval_latency 10              # where does the time go?
pytest -q                                    # offline tests
```

Scores are saved to `evals/results/`.

| Eval | Golden set | What it tells you |
|---|---|---|
| Retriever | `ideal_answer` | Contextual recall (did we miss needed info?) and precision (is the noise ranked low?) |
| Generator | `ideal_context` | Faithfulness, answer relevancy and citation format, with retrieval taken out of the picture |
| Pipeline | `ideal_answer` queries | RAG triad (context relevancy, faithfulness, answer relevancy) and the weakest link |
| Latency | `ideal_answer` queries | mean / p50 / p95 for dense retrieval, reranked retrieval, generation, end-to-end |

## The CI gate

1. Run the evals, then set the starting point:
   ```bash
   python -m evals.gate --update-baseline    # saves current scores to evals/baseline.json
   ```
   Commit `baseline.json`. Also review `evals/thresholds.json`: the minimums shipped there are **starting guesses**; set them a little below what your system actually scores.
2. From then on, `python -m evals.gate` fails (exit code 1) if any metric is below its minimum, **or** more than `max_drop` (default 0.05) below the baseline. The second check is what catches a slow slide that stays above the minimum.
3. In CI (`.github/workflows/eval.yml`) this runs on every pull request that touches `src/`, `evals/`, `goldens/` or `data/`, and the result table appears in the job summary.

Setup for CI: add `GROQ_API_KEY` as a repository secret and commit `data/apple_2024_report.pdf`. CI runs a sample of `EVAL_SAMPLE` goldens per suite (12 by default) to stay inside Groq rate limits.

When you improve the system on purpose, re-run the evals, run `--update-baseline`, and commit the new baseline with your change.

## Results

Fill this in after your first full run (don't commit numbers you haven't measured).

| Setup | Contextual recall | Contextual precision | RAG triad | p95 latency |
|---|---|---|---|---|
| Dense only | | | | |
| + Reranker | | | | |

## Known limitations

- The judge and the generator are both Groq-hosted LLMs from the same family, so the judge may be lenient on its own style. A different judge model is a stricter test (`JUDGE_MODEL` in `.env`).
- Citation accuracy checks that cited pages were in the context, not that the claim really sits on that page.
- CI uses a sample of the goldens, so small changes in score can be noise; run the full set locally before updating the baseline.
- Not yet built: correctness/completeness/style evaluation against the ideal answers, and safety evals (jailbreak, PII).
