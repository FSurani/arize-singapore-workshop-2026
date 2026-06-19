"""Lightweight retrieval over the Sunrise Outfitters knowledge base.

Builds a small in-memory vector store (``InMemoryVectorStore`` from
``langchain-core``) over the markdown docs in ``agent/knowledge/`` using Gemini
embeddings (``gemini-embedding-2``). It is tiny enough to build in a couple of
seconds on a free Colab runtime and needs no external vector database.

The store is built lazily and cached, so the first ``search_knowledge_base``
call pays the (small) embedding cost and later calls are instant.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
DEFAULT_K = 3


def _load_documents() -> list[Document]:
    """Load each knowledge-base markdown file as one ``Document``."""
    docs: list[Document] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


@lru_cache(maxsize=1)
def get_vector_store() -> InMemoryVectorStore:
    """Build (once) and return the in-memory vector store over the KB."""
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return InMemoryVectorStore.from_documents(_load_documents(), embedding=embeddings)


@lru_cache(maxsize=1)
def get_retriever():
    """Build (once) and return a LangChain retriever over the KB.

    We use a retriever (rather than calling ``similarity_search`` directly) so
    the OpenInference instrumentor emits a proper RETRIEVER span - the retrieved
    documents then show up in Arize and can be scored by retrieval evals.
    """
    return get_vector_store().as_retriever(search_kwargs={"k": DEFAULT_K})


def search_knowledge_base(query: str) -> str:
    """Return the knowledge-base passages most relevant to ``query``."""
    results = get_retriever().invoke(query)
    if not results:
        return "No relevant policy found in the knowledge base."
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('source', 'kb')}]\n{doc.page_content}" for doc in results
    )
