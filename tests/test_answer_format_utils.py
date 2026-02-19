from __future__ import annotations

import unittest

import pandas as pd

from src.answer_format_utils import (
    format_answer_text,
    looks_like_structured_answer,
    summarize_table_for_answer,
)


class AnswerFormatUtilsTests(unittest.TestCase):
    def test_looks_like_structured_answer_detects_list_of_dicts(self) -> None:
        self.assertTrue(looks_like_structured_answer("[{'industry':'Finance','net_sales':1000}]"))

    def test_looks_like_structured_answer_false_for_plain_text(self) -> None:
        self.assertFalse(looks_like_structured_answer("Finance leads net sales this month."))

    def test_summarize_table_for_answer_mentions_row_count(self) -> None:
        frame = pd.DataFrame(
            {
                "industry": ["Finance", "Retail"],
                "net_sales": [1000.0, 400.0],
                "profit": [200.0, 50.0],
            }
        )
        summary = summarize_table_for_answer("show industry sales", frame)
        self.assertIn("2 rows", summary)
        self.assertIn("highest", summary.lower())

    def test_format_answer_text_rewrites_structured_answer(self) -> None:
        frame = pd.DataFrame(
            {
                "industry": ["Finance", "Retail"],
                "net_sales": [1000.0, 400.0],
            }
        )
        formatted = format_answer_text(
            "[{'industry':'Finance','net_sales':1000}]",
            question="show industry sales",
            computed_df=frame,
        )
        self.assertIn("**Top Insights**", formatted)
        self.assertIn("**Key Takeaway:**", formatted)
        self.assertGreaterEqual(formatted.count("\n- "), 3)

    def test_format_answer_text_rewrites_plain_text_into_template(self) -> None:
        frame = pd.DataFrame(
            {
                "industry": ["Finance", "Retail"],
                "net_sales": [1000.0, 400.0],
            }
        )
        formatted = format_answer_text(
            "Finance appears to lead overall.",
            question="show industry sales",
            computed_df=frame,
        )
        self.assertTrue(formatted.startswith("**Top Insights**"))
        self.assertIn("**Key Takeaway:**", formatted)


if __name__ == "__main__":
    unittest.main()
