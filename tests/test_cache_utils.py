from __future__ import annotations

import unittest

from src.cache_utils import cosine_similarity, normalize_question


class CacheUtilsTests(unittest.TestCase):
    def test_normalize_question_removes_punctuation_and_case(self) -> None:
        normalized = normalize_question("  Which CUSTOMER spent the most?  ")
        self.assertEqual(normalized, "which customer spent the most")

    def test_cosine_similarity_identical_vectors(self) -> None:
        score = cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_cosine_similarity_orthogonal_vectors(self) -> None:
        score = cosine_similarity([1.0, 0.0], [0.0, 1.0])
        self.assertAlmostEqual(score, 0.0, places=6)

    def test_cosine_similarity_handles_empty_input(self) -> None:
        score = cosine_similarity([], [1.0, 2.0])
        self.assertEqual(score, -1.0)


if __name__ == "__main__":
    unittest.main()
