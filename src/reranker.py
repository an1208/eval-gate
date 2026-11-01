"""
Reranking retriever: over-retrieve with the fast bi-encoder, then re-score each
(query, chunk) pair TOGETHER with a cross-encoder and keep the best TOP_K.

It has the same `.invoke(query)` interface as the dense retriever, so the
pipeline and the evals can swap one for the other.

    python -m src.reranker "What were Apple's iPhone net sales in fiscal 2024?"
"""

import sys

from langsmith import traceable
from sentence_transformers import CrossEncoder

from src.config import FETCH_K, RERANK_MODEL, TOP_K
from src.retriever import build_vectorstore


class RerankingRetriever:
    def __init__(self, fetch_k=FETCH_K, top_k=TOP_K):
        self.store = build_vectorstore()
        self.reranker = CrossEncoder(RERANK_MODEL)  # downloads once (~80MB) on first run
        self.fetch_k = fetch_k   # how many the bi-encoder brings back (over-retrieve)
        self.top_k = top_k       # how many survive after reranking

    @traceable(run_type="retriever", name="RerankingRetriever")
    def invoke(self, query):
        # 1. OVER-RETRIEVE: fast bi-encoder, deliberately more than we need
        candidates = self.store.similarity_search(query, k=self.fetch_k)

        # 2. RERANK: score each (query, chunk) pair together with the cross-encoder
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)

        # 3. SORT by score, keep the best top_k
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_k]]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What were total net sales in fiscal 2024?"
    for i, d in enumerate(RerankingRetriever().invoke(question), 1):
        print(f"\n--- Chunk {i} (page {d.metadata.get('page', 0) + 1}) ---")
        print(d.page_content[:300])
