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

from src.config import GROQ_MAX_RETRIES, LLM_MODEL

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


# Models sometimes cite with full-width brackets, [p.12] or [ p. 32, p. 26 ]; rewrite every variant to
# exactly [p. 12] / [p. 32, 26] so anything that parses citations downstream (and the
# evals) sees one format. Brackets holding anything other than page numbers are left alone.
_CITATION_BRACKET = re.compile(r"[\[\u3010\u3014]\s*(p(?:ages?|p)?\b[^\]\u3011\u3015\[]*)[\]\u3011\u3015]", re.IGNORECASE)


def normalize_citations(text: str) -> str:
    def fix(match):
        body = match.group(1)
        if re.search(r"[A-Za-z]", re.sub(r"\b(?:pages?|pp?)\b", "", body, flags=re.IGNORECASE)):
            return match.group(0)  # contains other words, so it isn't a page citation
        pages = re.findall(r"\d+(?:\s*[-\u2013]\s*\d+)?", body)
        if not pages:
            return match.group(0)
        return "[p. " + ", ".join(re.sub(r"\s+", "", page) for page in pages) + "]"

    return _CITATION_BRACKET.sub(fix, text)


def format_docs(docs):
    """Step 3 - join chunks into one context string, labelled with page numbers."""
    return "\n\n".join(f"[p. {d.metadata.get('page', 0) + 1}] {d.page_content}" for d in docs)


@lru_cache(maxsize=1)
def get_generator_chain():
    """Step 4 - prompt -> LLM -> plain text -> tidy citations. Expects {"context": ..., "question": ...}."""
    return (
        PROMPT
        | ChatGroq(model=LLM_MODEL, temperature=0, max_retries=GROQ_MAX_RETRIES)
        | StrOutputParser()
        | RunnableLambda(normalize_citations)
    )


def generate(question: str, docs: list) -> str:
    """Answer a question from a list of retrieved Documents."""
    return get_generator_chain().invoke({"question": question, "context": format_docs(docs)})


if __name__ == "__main__":
    test_docs = [Document(page_content="Total net sales were $391,035 million in fiscal 2024.", metadata={"page": 0})]
    print(generate("What were total net sales in fiscal 2024?", test_docs))