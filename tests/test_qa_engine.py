from __future__ import annotations

import unittest

from src.qa_engine import answer_with_context


class _FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.response


class QAEngineTests(unittest.TestCase):
    def test_answer_with_context_parses_json_output(self) -> None:
        llm = _FakeLLMClient(
            '{"answer": "Top segment is Enterprise.", "reasoning_summary": "Used SQL totals grouped by segment."}'
        )
        result = answer_with_context(
            question="Which segment has the highest net sales?",
            overview={"row_count": 100, "column_count": 5, "missing_values": 0, "duplicate_rows": 0},
            schema_summary_text="segment, net_sales",
            dictionary_text="segment definition",
            retrieved_documents=[],
            computed_result_text="segment,total\nEnterprise,1000",
            generated_sql="SELECT segment, SUM(net_sales) FROM data GROUP BY 1",
            llm_client=llm,
        )
        self.assertEqual(result.answer, "Top segment is Enterprise.")
        self.assertIn("SQL", result.reasoning_summary)

    def test_answer_with_context_parses_fenced_json_output(self) -> None:
        llm = _FakeLLMClient(
            """```json
{"answer":"A","reasoning_summary":"B"}
```"""
        )
        result = answer_with_context(
            question="q",
            overview={"row_count": 1, "column_count": 1, "missing_values": 0, "duplicate_rows": 0},
            schema_summary_text="x",
            dictionary_text="",
            retrieved_documents=[],
            computed_result_text="",
            generated_sql="",
            llm_client=llm,
        )
        self.assertEqual(result.answer, "A")
        self.assertEqual(result.reasoning_summary, "B")

    def test_answer_with_context_falls_back_for_unstructured_output(self) -> None:
        llm = _FakeLLMClient("Plain text answer only")
        result = answer_with_context(
            question="q",
            overview={"row_count": 1, "column_count": 1, "missing_values": 0, "duplicate_rows": 0},
            schema_summary_text="x",
            dictionary_text="",
            retrieved_documents=[],
            computed_result_text="",
            generated_sql="",
            llm_client=llm,
        )
        self.assertEqual(result.answer, "Plain text answer only")
        self.assertIn("unstructured", result.reasoning_summary.lower())


if __name__ == "__main__":
    unittest.main()
