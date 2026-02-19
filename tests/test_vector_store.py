from __future__ import annotations

import unittest
from unittest.mock import patch

from src.vector_store import (
    HashingEmbeddingFunction,
    OllamaEmbeddingFunction,
    OpenAIEmbeddingFunction,
    index_document_batches,
    query_collection,
)


class _FakeCollection:
    class _FakeClient:
        def __init__(self, max_batch_size: int) -> None:
            self.max_batch_size = max_batch_size

        def get_max_batch_size(self) -> int:
            return self.max_batch_size

    def __init__(self, max_batch_size: int | None = None) -> None:
        self.add_calls: list[tuple[list[str], list[dict[str, int]], list[str]]] = []
        if max_batch_size is not None:
            self._client = self._FakeClient(max_batch_size)

    def add(
        self,
        *,
        documents: list[str],
        metadatas: list[dict[str, int]],
        ids: list[str],
    ) -> None:
        self.add_calls.append((documents, metadatas, ids))

    def query(self, **_: object) -> dict[str, list[list[object]]]:
        return {
            "documents": [["doc-a", "doc-b"]],
            "metadatas": [[{"row_position": 0}, {"row_position": 1}]],
            "distances": [[0.1, 0.2]],
        }


class VectorStoreTests(unittest.TestCase):
    def test_index_document_batches_counts_rows(self) -> None:
        collection = _FakeCollection()
        progress_counts: list[int] = []
        batches = [
            (
                ["row1", "row2"],
                [{"row_position": 0}, {"row_position": 1}],
                ["row_0", "row_1"],
            ),
            (
                ["row3"],
                [{"row_position": 2}],
                ["row_2"],
            ),
        ]

        indexed = index_document_batches(
            collection,
            batches=batches,
            on_batch_indexed=lambda count: progress_counts.append(count),
        )
        self.assertEqual(indexed, 3)
        self.assertEqual(progress_counts, [2, 3])
        self.assertEqual(len(collection.add_calls), 2)

    def test_index_document_batches_respects_collection_max_batch_size(self) -> None:
        collection = _FakeCollection(max_batch_size=2)
        progress_counts: list[int] = []
        batches = [
            (
                ["row1", "row2", "row3", "row4", "row5"],
                [
                    {"row_position": 0},
                    {"row_position": 1},
                    {"row_position": 2},
                    {"row_position": 3},
                    {"row_position": 4},
                ],
                ["row_0", "row_1", "row_2", "row_3", "row_4"],
            ),
        ]

        indexed = index_document_batches(
            collection,
            batches=batches,
            on_batch_indexed=lambda count: progress_counts.append(count),
        )

        self.assertEqual(indexed, 5)
        self.assertEqual(progress_counts, [2, 4, 5])
        self.assertEqual(len(collection.add_calls), 3)
        self.assertEqual(collection.add_calls[0][0], ["row1", "row2"])
        self.assertEqual(collection.add_calls[1][0], ["row3", "row4"])
        self.assertEqual(collection.add_calls[2][0], ["row5"])

    def test_query_collection_extracts_first_result_set(self) -> None:
        collection = _FakeCollection()
        docs, metadatas, distances = query_collection(collection, "top segment", top_k=2)
        self.assertEqual(docs, ["doc-a", "doc-b"])
        self.assertEqual(metadatas[0]["row_position"], 0)
        self.assertEqual(distances, [0.1, 0.2])

    def test_embedding_functions_expose_name_contract(self) -> None:
        hashing = HashingEmbeddingFunction(n_features=128)
        self.assertIsInstance(hashing.name(), str)
        self.assertTrue(hashing.name())

        ollama = OllamaEmbeddingFunction(
            base_url="http://localhost:11434",
            model="mxbai-embed-large",
        )
        self.assertIsInstance(ollama.name(), str)
        self.assertTrue(ollama.name())

        openai = OpenAIEmbeddingFunction.__new__(OpenAIEmbeddingFunction)
        openai.model = "text-embedding-3-small"
        openai.timeout_seconds = 120
        self.assertIsInstance(openai.name(), str)
        self.assertTrue(openai.name())

    @patch("src.vector_store.requests.post")
    def test_ollama_embedding_function_uses_batch_embed_endpoint(self, mock_post) -> None:
        class _Response:
            def __init__(self, embeddings: list[list[float]]) -> None:
                self._embeddings = embeddings

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {"embeddings": self._embeddings}

        def _mock_post(*_: object, **kwargs: object) -> _Response:
            payload = kwargs.get("json", {})
            if isinstance(payload, dict):
                batch_input = payload.get("input", [])
                if isinstance(batch_input, list):
                    embeddings = [[float(i), float(i) + 0.1] for i, _ in enumerate(batch_input, 1)]
                    return _Response(embeddings)
            return _Response([[0.1, 0.2]])

        mock_post.side_effect = _mock_post
        embedder = OllamaEmbeddingFunction(
            base_url="http://localhost:11434",
            model="mxbai-embed-large",
        )

        vectors = embedder(["hello", "world"])
        self.assertEqual(vectors, [[1.0, 1.1], [2.0, 2.1]])
        self.assertEqual(mock_post.call_count, 1)

        query_vectors = embedder.embed_query("hello")
        self.assertEqual(query_vectors, [[1.0, 1.1]])
        self.assertEqual(mock_post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
