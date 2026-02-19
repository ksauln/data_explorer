from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import chromadb
import numpy as np
import requests
from chromadb.api.models.Collection import Collection
from sklearn.feature_extraction.text import HashingVectorizer


class HashingEmbeddingFunction:
    """Fully local embedding function with deterministic vectors."""

    def __init__(self, n_features: int = 512):
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm=None,
        )

    def __call__(self, input: list[str]) -> list[list[float]]:
        matrix = self.vectorizer.transform(input)
        vectors = matrix.astype(np.float32).toarray()
        return [[float(value) for value in row] for row in vectors.tolist()]

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    def embed_query(self, input: list[str] | str) -> list[list[float]]:
        texts = [input] if isinstance(input, str) else list(input)
        return self.__call__(texts)

    def name(self) -> str:
        return "hashing-embedding-function"

    def get_config(self) -> dict[str, int]:
        return {"n_features": int(self.vectorizer.n_features)}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "HashingEmbeddingFunction":
        return HashingEmbeddingFunction(n_features=int(config.get("n_features", 512)))


class OpenAIEmbeddingFunction:
    def __init__(self, api_key: str, model: str, timeout_seconds: int = 120):
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)

    def __call__(self, input: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=list(input))
        ordered = sorted(response.data, key=lambda item: item.index)
        return [[float(value) for value in item.embedding] for item in ordered]

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    def embed_query(self, input: list[str] | str) -> list[list[float]]:
        texts = [input] if isinstance(input, str) else list(input)
        return self.__call__(texts)

    def name(self) -> str:
        return "openai-embedding-function"

    def get_config(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "timeout_seconds": int(self.timeout_seconds),
        }

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "OpenAIEmbeddingFunction":
        raise NotImplementedError(
            "OpenAIEmbeddingFunction.build_from_config is not supported without API key injection."
        )


class OllamaEmbeddingFunction:
    def __init__(self, base_url: str, model: str, timeout_seconds: int = 120):
        self.base_url = base_url
        self.model = model
        self.timeout_seconds = timeout_seconds

    def __call__(self, input: list[str]) -> list[list[float]]:
        texts = list(input)
        if not texts:
            return []

        base = self.base_url.rstrip("/")
        batch_endpoint = f"{base}/api/embed"
        single_endpoint = f"{base}/api/embeddings"

        try:
            response = requests.post(
                batch_endpoint,
                json={"model": self.model, "input": texts},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings")
            if isinstance(embeddings, list):
                return _normalize_embeddings(embeddings)
        except Exception:
            pass

        embeddings: list[list[float]] = []
        for text in texts:
            response = requests.post(
                single_endpoint,
                json={"model": self.model, "prompt": text},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            vector = data.get("embedding")
            if not isinstance(vector, list):
                raise ValueError("Invalid Ollama embedding response format.")
            embeddings.append([float(value) for value in vector])

        return embeddings

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    def embed_query(self, input: list[str] | str) -> list[list[float]]:
        texts = [input] if isinstance(input, str) else list(input)
        return self.__call__(texts)

    def name(self) -> str:
        return "ollama-embedding-function"

    def get_config(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "timeout_seconds": int(self.timeout_seconds),
        }

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "OllamaEmbeddingFunction":
        return OllamaEmbeddingFunction(
            base_url=str(config.get("base_url", "http://localhost:11434")),
            model=str(config.get("model", "mxbai-embed-large")),
            timeout_seconds=int(config.get("timeout_seconds", 120)),
        )


def get_chroma_client(persist_directory: Path) -> chromadb.PersistentClient:
    persist_directory.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_directory))


def rebuild_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    embedding_function: Any,
) -> Collection:
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )


def get_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    embedding_function: Any,
) -> Collection:
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )


def index_documents(
    collection: Collection,
    documents: list[str],
    metadatas: list[dict[str, Any]],
    ids: list[str],
    batch_size: int = 500,
) -> int:
    if not documents:
        return 0

    safe_batch_size = max(1, int(batch_size))
    max_batch_size = _get_collection_max_batch_size(collection)
    if max_batch_size is not None:
        safe_batch_size = min(safe_batch_size, max_batch_size)

    for start in range(0, len(documents), safe_batch_size):
        end = start + safe_batch_size
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )

    return len(documents)


def index_document_batches(
    collection: Collection,
    batches: Iterable[tuple[list[str], list[dict[str, Any]], list[str]]],
    on_batch_indexed: Callable[[int], None] | None = None,
) -> int:
    indexed_count = 0
    max_batch_size = _get_collection_max_batch_size(collection)
    for documents, metadatas, ids in batches:
        if not documents:
            continue
        chunk_size = len(documents) if max_batch_size is None else max_batch_size
        for start in range(0, len(documents), chunk_size):
            end = start + chunk_size
            collection.add(
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )
            indexed_count += len(documents[start:end])
            if on_batch_indexed is not None:
                on_batch_indexed(indexed_count)
    return indexed_count


def query_collection(
    collection: Collection,
    query_text: str,
    top_k: int = 5,
) -> tuple[list[str], list[dict[str, Any]], list[float]]:
    result = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return documents, metadatas, distances


def _normalize_embeddings(embeddings: list[Any]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for row in embeddings:
        values = row.tolist() if hasattr(row, "tolist") else row
        if not isinstance(values, list):
            raise ValueError("Invalid embedding row format returned by provider.")
        normalized.append([float(value) for value in values])
    return normalized


def _get_collection_max_batch_size(collection: Collection) -> int | None:
    client = getattr(collection, "_client", None)
    if client is None:
        return None

    get_max_batch_size = getattr(client, "get_max_batch_size", None)
    if not callable(get_max_batch_size):
        return None

    try:
        max_batch_size = int(get_max_batch_size())
    except Exception:
        return None

    return max_batch_size if max_batch_size > 0 else None
