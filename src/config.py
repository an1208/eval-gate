"""
Central configuration. Every tunable lives here; .env is loaded on import.
Change a value here (or via .env) and the pipeline AND the evals pick it up.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ---- Data ----
PDF_PATH = os.getenv("PDF_PATH", str(ROOT / "data" / "apple_2024_report.pdf"))
CHROMA_DIR = os.getenv("CHROMA_DIR", str(ROOT / "chroma_db"))

# ---- Indexing ----
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---- Retrieval ----
TOP_K = 4                 # chunks that reach the generator
FETCH_K = 10              # chunks the reranker chooses from (over-retrieve)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() in ("1", "true", "yes")

# ---- Generation + judging (both on Groq) ----
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "6"))  # retries on rate limits (the latency eval sets 0)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")
# gpt-oss models "think" before answering; less thinking = shorter, faster, cheaper judge calls
# and fewer replies cut off mid-JSON. Use "medium" or "high" for a more careful judge.
JUDGE_REASONING_EFFORT = os.getenv("JUDGE_REASONING_EFFORT", "low")