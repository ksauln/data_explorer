from __future__ import annotations

import re
from typing import Any

import pandas as pd


def looks_like_structured_answer(answer_text: str) -> bool:
    text = answer_text.strip()
    if not text:
        return False

    if (
        (text.startswith("[{") and text.endswith("}]"))
        or (text.startswith("{") and text.endswith("}"))
        or (text.startswith("[") and text.endswith("]"))
    ) and ":" in text:
        return True

    if re.search(r"\{'[^']+'\s*:", text):
        return True
    if re.search(r'"[^"]+"\s*:', text):
        return True
    return False


def format_answer_text(
    answer_text: str,
    *,
    question: str,
    computed_df: pd.DataFrame,
) -> str:
    cleaned = answer_text.strip()
    if _looks_like_strict_template(cleaned):
        return cleaned

    if not cleaned:
        cleaned = "No answer text returned."

    if looks_like_structured_answer(cleaned):
        cleaned = summarize_table_for_answer(question, computed_df)

    insights = _build_top_insights(
        question=question,
        computed_df=computed_df,
        narrative=cleaned,
    )
    key_takeaway = _build_key_takeaway(
        computed_df=computed_df,
        insights=insights,
        fallback=cleaned,
    )
    bullets = "\n".join([f"- {insight}" for insight in insights[:3]])
    return f"**Top Insights**\n{bullets}\n\n**Key Takeaway:** {key_takeaway}"


def summarize_table_for_answer(question: str, computed_df: pd.DataFrame) -> str:
    if computed_df.empty:
        return "I could not find rows for that request."

    columns = list(computed_df.columns)
    preview_columns = ", ".join(str(column) for column in columns[:4])
    if len(columns) > 4:
        preview_columns += ", ..."

    summary = (
        f"I processed your request and found {len(computed_df):,} rows. "
        f"The result table below includes: {preview_columns}."
    )

    top_stat = _top_metric_statement(computed_df)
    if top_stat:
        summary += f" {top_stat}"
    return summary


def _looks_like_strict_template(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized.startswith("**top insights**") and "**key takeaway:**" in normalized


def _build_top_insights(
    *,
    question: str,
    computed_df: pd.DataFrame,
    narrative: str,
) -> list[str]:
    insights: list[str] = []
    first_sentence = _first_sentence(narrative)
    if first_sentence:
        insights.append(first_sentence)

    if computed_df.empty:
        insights.append("No computed table was returned for this question.")
        insights.append("Try a narrower prompt with explicit grouping or filters if needed.")
        return _exactly_three(insights)

    insights.append(
        f"The result table contains {len(computed_df):,} rows across {computed_df.shape[1]} columns."
    )

    top_stat = _top_metric_statement(computed_df)
    if top_stat:
        insights.append(top_stat)

    if len(insights) < 3:
        numeric_columns = computed_df.select_dtypes(include="number").columns.tolist()
        if numeric_columns:
            metric = str(numeric_columns[0])
            metric_total = float(computed_df[metric].fillna(0).sum())
            insights.append(f"The total {metric} across returned rows is {metric_total:,.2f}.")

    if len(insights) < 3:
        insights.append(f"This answer is based on the computed SQL result for: {question.strip()}")

    return _exactly_three(insights)


def _build_key_takeaway(
    *,
    computed_df: pd.DataFrame,
    insights: list[str],
    fallback: str,
) -> str:
    top_stat = _top_metric_statement(computed_df)
    if top_stat:
        return top_stat
    if insights:
        return insights[0]
    return _first_sentence(fallback) or "Review the table below for full details."


def _top_metric_statement(computed_df: pd.DataFrame) -> str:
    if computed_df.empty:
        return ""

    numeric_columns = computed_df.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        return ""

    metric_column = str(numeric_columns[0])
    key_candidates = [column for column in computed_df.columns if str(column) != metric_column]
    key_column = str(key_candidates[0]) if key_candidates else metric_column

    ranked = computed_df[[key_column, metric_column]].dropna()
    if ranked.empty:
        return ""

    top_row = ranked.sort_values(metric_column, ascending=False).head(1).iloc[0]
    key_value = _as_text(top_row[key_column])
    metric_value = float(top_row[metric_column])
    return f"The highest {metric_column} is for {key_column} '{key_value}' at {metric_value:,.2f}."


def _first_sentence(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)
    return parts[0].strip()


def _exactly_three(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(normalized)

    while len(deduped) < 3:
        deduped.append("See the result table below for the detailed breakdown.")
    return deduped[:3]


def _as_text(value: Any) -> str:
    if pd.isna(value):
        return "NULL"
    return str(value)
