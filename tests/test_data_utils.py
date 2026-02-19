from __future__ import annotations

import unittest

import pandas as pd

from src.data_utils import (
    dashboard_sample,
    iter_dataframe_document_batches,
    load_reference_text,
    normalize_dataframe,
    schema_summary,
)


class DataUtilsTests(unittest.TestCase):
    def test_dashboard_sample_caps_rows(self) -> None:
        frame = pd.DataFrame({"id": range(1000)})
        sampled = dashboard_sample(frame, max_rows=100)
        self.assertEqual(len(sampled), 100)

    def test_load_reference_text_from_txt(self) -> None:
        text = load_reference_text("dictionary.txt", b"join_key=customer_id\nfact=sales")
        self.assertIn("join_key=customer_id", text)

    def test_iter_dataframe_document_batches_batches_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "segment": ["A", "B", "C"],
                "revenue": [100.0, 200.5, 300.25],
            }
        )
        batches = list(
            iter_dataframe_document_batches(
                frame,
                max_rows=3,
                batch_size=2,
                max_columns=2,
            )
        )
        self.assertEqual(len(batches), 2)
        first_docs, _, first_ids = batches[0]
        self.assertEqual(len(first_docs), 2)
        self.assertEqual(first_ids[0], "row_0")
        self.assertIn("segment=A", first_docs[0])

    def test_schema_summary_contains_column_name(self) -> None:
        frame = pd.DataFrame({"revenue": [1, 2, 3], "segment": ["A", "B", "A"]})
        summary = schema_summary(frame, max_profile_rows=100)
        self.assertIn("- revenue", summary)
        self.assertIn("- segment", summary)

    def test_normalize_dataframe_parses_likely_date_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "sale_date": ["2025-01-01", "2025-01-15", "2025-02-01"],
                "value": [1, 2, 3],
            }
        )
        normalized = normalize_dataframe(frame)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(normalized["sale_date"]))


if __name__ == "__main__":
    unittest.main()
