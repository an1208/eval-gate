"""
The full RAG chain: question -> retrieve -> augment -> generate.

    python -m src.pipeline "What were Apple's total net sales in fiscal 2024?"
    python -m src.pipeline            # interactive mode

USE_RERANKER (config / .env) switches between the reranking and dense retriever.
"""

import sys
from functools import lru_cache

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.config import USE_RERANKER
from src.generator import format_docs, get_generator_chain
from src.retriever import get_retriever


def build_retriever(use_reranker: bool = USE_RERANKER):
    if use_reranker:
        from src.reranker import RerankingRetriever

        return RerankingRetriever()
    return get_retriever()


@lru_cache(maxsize=1)
def _chain():
    retriever = build_retriever()
    # A dict flows through; each stage adds a key:
    # {question} -> {question, docs} -> {question, docs, context} -> {..., answer}
    return (
        RunnablePassthrough.assign(docs=RunnableLambda(lambda x: retriever.invoke(x["question"])))
        | RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | RunnablePassthrough.assign(answer=get_generator_chain())
    )


def ask(question: str) -> dict:
    out = _chain().invoke({"question": question})
    pages = sorted({d.metadata.get("page", 0) + 1 for d in out["docs"]})
    return {"answer": out["answer"], "pages": pages, "docs": out["docs"]}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        questions = iter(lambda: input("\nQuestion (blank to quit): ").strip(), "")

    for q in questions:
        result = ask(q)
        print(f"\n{result['answer']}\n\nSources: pages {result['pages']}")
