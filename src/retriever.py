"""
Indexing + dense retrieval.

    Step 1a  load the PDF            Step 1c/1d  embed + store in Chroma
    Step 1b  split into chunks       Step 2      retrieve top-k

Try it on its own:
    python -m src.retriever "What were Apple's iPhone net sales in fiscal 2024?"
    python -m src.retriever --reindex          # rebuild the index from the PDF
"""

import shutil
import sys
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, PDF_PATH, TOP_K


def load_documents():
    """Step 1a - read the PDF; each page becomes a Document with page metadata."""
    if not Path(PDF_PATH).is_file():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}. Put your report there or set PDF_PATH in .env.")
    return PyPDFLoader(PDF_PATH).load()


def split_documents(pages):
    """Step 1b - break pages into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(pages)


@lru_cache(maxsize=1)
def build_vectorstore():
    """Step 1c & 1d - embed chunks and store them in Chroma (built once, then reused)."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if Path(CHROMA_DIR).is_dir() and any(Path(CHROMA_DIR).iterdir()):
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    pages = load_documents()
    chunks = split_documents(pages)
    store = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    print(f"Indexed {len(chunks)} chunks from {len(pages)} pages.")
    return store


def reindex():
    """Delete the saved index and rebuild it (use after changing chunking or embeddings)."""
    build_vectorstore.cache_clear()
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    return build_vectorstore()


def get_retriever():
    """Step 2 - a retriever returning the top-k most similar chunks."""
    return build_vectorstore().as_retriever(search_kwargs={"k": TOP_K})


if __name__ == "__main__":
    if "--reindex" in sys.argv:
        reindex()
        sys.exit(0)

    question = " ".join(sys.argv[1:]) or "What were total net sales in fiscal 2024?"
    for i, d in enumerate(get_retriever().invoke(question), 1):
        print(f"\n--- Chunk {i} (page {d.metadata.get('page', 0) + 1}) ---")
        print(d.page_content[:300])
