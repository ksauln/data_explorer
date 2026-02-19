from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.llm_clients import LLMClient


SYSTEM_PROMPT = """
You are a data analyst assistant answering questions about a user-uploaded dataset.
Use only the provided dataset context, SQL results, and retrieved rows.
If information is missing or uncertain, say so clearly.
When SQL results are provided, treat them as the highest-confidence evidence.
When useful, provide concise calculations and assumptions.
Output valid JSON with keys "answer" and "reasoning_summary".
The "reasoning_summary" must be concise diagnostic notes, not hidden chain-of-thought.
"""


@dataclass
class QAResponse:
    answer: str
    reasoning_summary: str
    raw_model_output: str


def answer_with_context(
    question: str,
    *,
    overview: dict[str, float | int],
    schema_summary_text: str,
    dictionary_text: str,
    retrieved_documents: list[str],
    computed_result_text: str,
    generated_sql: str,
    llm_client: LLMClient,
) -> QAResponse:
    rows_context = "\n".join(
        [f"Row snippet {idx + 1}: {doc}" for idx, doc in enumerate(retrieved_documents)]
    )

    user_prompt = f"""
Dataset overview:
- rows={overview.get("row_count")}
- columns={overview.get("column_count")}
- missing_values={overview.get("missing_values")}
- duplicate_rows={overview.get("duplicate_rows")}

Schema summary:
{schema_summary_text}

Retrieved row context:
{rows_context}

Generated SQL (if available):
{generated_sql if generated_sql.strip() else "No SQL query generated."}

Computed query result preview:
{computed_result_text if computed_result_text.strip() else "No computed result available."}

Data dictionary / join-key context:
{dictionary_text if dictionary_text.strip() else "No data dictionary provided."}

User question:
{question}

Answer requirements:
- Ground your answer in the dataset context above.
- Prioritize SQL results when they are available and relevant.
- If exact values are not available, state the limitation.
- Keep response concise and actionable.
Output format:
- Return only JSON.
- JSON schema:
  {{
    "answer": "<final answer for the user>",
    "reasoning_summary": "<2-6 concise bullets/sentences describing data used, method, and limits>"
  }}
""".strip()

    model_output = llm_client.generate(SYSTEM_PROMPT.strip(), user_prompt)
    return _parse_model_output(model_output)


def _parse_model_output(model_output: str) -> QAResponse:
    payload = _extract_json_payload(model_output)
    if payload is None:
        return QAResponse(
            answer=model_output.strip() or "No response returned by the model.",
            reasoning_summary=(
                "Model returned unstructured output; no diagnostic reasoning summary available."
            ),
            raw_model_output=model_output,
        )

    answer = str(payload.get("answer", "")).strip()
    reasoning_summary = str(payload.get("reasoning_summary", "")).strip()
    if not answer:
        answer = "No answer returned by the model."
    if not reasoning_summary:
        reasoning_summary = "Model did not provide a reasoning summary."

    return QAResponse(
        answer=answer,
        reasoning_summary=reasoning_summary,
        raw_model_output=model_output,
    )


def _extract_json_payload(model_output: str) -> dict[str, object] | None:
    text = model_output.strip()
    if not text:
        return None

    candidates: list[str] = [text]
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        candidates.insert(0, fenced_match.group(1).strip())

    brace_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(1).strip())

    for candidate in candidates:
        try:
            decoded = json.loads(candidate)
        except Exception:
            continue
        if isinstance(decoded, dict):
            return decoded
    return None
