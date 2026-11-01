"""
Augmentation + generation.

Takes a question and retrieved chunks, builds the prompt (cite pages, say
"I don't know" when the context lacks the answer) and asks the LLM. It does
not depend on the retriever, so it can be tested with hand-written chunks.

    python -m src.generator
"""

import re
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq

from src.config import LLM_MODEL

REFUSAL = "I don't know based on the provided document."

PROMPT = ChatPromptTemplate.from_template(
    """You are an assistant answering questions about the provided document.
Use ONLY the context below. After each claim, cite the page number(s) in plain square brackets,
exactly like [p. 12] or [p. 12, 14]. Never use any other bracket style.
If the context does not contain the answer, reply exactly: \"""" + REFUSAL + """\"

Context:
{context}

Question: {question}

Answer:"""
)


# Models sometimes cite as 【p. 12】 or [p.12]; rewrite every variant to exactly [p. 12]
# so anything that parses citations downstream (and the evals) sees one format.
_CITATION_VARIANT = re.compile(r"[\[【〔]\s*p(?:ages?)?\.?\s*([\d,\s\-–]+?)\s*[\]】〕]", re.IGNORECASE)


def normalize_citations(text: str) -> str:
    return _CITATION_VARIANT.sub(lambda m: "[p. " + re.sub(r"\s*,\s*", ", ", m.group(1).strip()) + "]", text)


def format_docs(docs):
    """Step 3 - join chunks into one context string, labelled with page numbers."""
    return "\n\n".join(f"[p. {d.metadata.get('page', 0) + 1}] {d.page_content}" for d in docs)


@lru_cache(maxsize=1)
def get_generator_chain():
    """Step 4 - prompt -> LLM -> plain text -> tidy citations. Expects {"context": ..., "question": ...}."""
    return (
        PROMPT
        | ChatGroq(model=LLM_MODEL, temperature=0, max_retries=6)
        | StrOutputParser()
        | RunnableLambda(normalize_citations)
    )


def generate(question: str, docs: list) -> str:
    """Answer a question from a list of retrieved Documents."""
    return get_generator_chain().invoke({"question": question, "context": format_docs(docs)})


if __name__ == "__main__":
    test_docs = [Document(page_content="Total net sales were $391,035 million in fiscal 2024.", metadata={"page": 0})]
    print(generate("What were total net sales in fiscal 2024?", test_docs))