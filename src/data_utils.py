from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls", ".json", ".parquet"}
REFERENCE_SUFFIXES = {".txt", ".md", ".csv", ".xlsx", ".xls", ".json"}


def load_dataframe(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type: {suffix}. Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )

    data_buffer = io.BytesIO(file_bytes)

    if suffix == ".csv":
        df = pd.read_csv(data_buffer, low_memory=False)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(data_buffer)
    elif suffix == ".json":
        df = pd.read_json(data_buffer)
    elif suffix == ".parquet":
        df = pd.read_parquet(data_buffer)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return normalize_dataframe(df)


def load_reference_text(
    file_name: str,
    file_bytes: bytes,
    max_chars: int = 60_000,
) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix not in REFERENCE_SUFFIXES:
        raise ValueError(
            f"Unsupported reference file type: {suffix}. Supported: {sorted(REFERENCE_SUFFIXES)}"
        )

    data_buffer = io.BytesIO(file_bytes)
    if suffix in {".txt", ".md"}:
        text = data_buffer.getvalue().decode("utf-8", errors="replace")
        return text[:max_chars]

    if suffix == ".csv":
        frame = pd.read_csv(data_buffer, low_memory=False).head(2000)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(data_buffer).head(2000)
    elif suffix == ".json":
        frame = pd.read_json(data_buffer).head(2000)
    else:
        raise ValueError(f"Unsupported reference file type: {suffix}")

    as_text = frame.to_csv(index=False)
    return as_text[:max_chars]


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    _parse_likely_datetime_columns(normalized)
    return normalized


def normalize_table_name(file_name: str) -> str:
    stem = Path(file_name).stem.lower()
    cleaned = re.sub(r"[^a-z0-9_]+", "_", stem)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "table"
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned


def dataset_overview(df: pd.DataFrame) -> dict[str, float | int]:
    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
    }


def dashboard_sample(df: pd.DataFrame, max_rows: int = 200_000) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)


def schema_summary(
    df: pd.DataFrame,
    max_categories: int = 6,
    max_profile_rows: int = 200_000,
) -> str:
    profile_frame = dashboard_sample(df, max_rows=max_profile_rows)
    lines: list[str] = []
    for column in profile_frame.columns:
        dtype = str(profile_frame[column].dtype)
        missing = int(profile_frame[column].isna().sum())
        unique_count = int(profile_frame[column].nunique(dropna=True))

        if pd.api.types.is_numeric_dtype(profile_frame[column]):
            min_value = _to_serializable(profile_frame[column].min())
            max_value = _to_serializable(profile_frame[column].max())
            mean_value = _to_serializable(profile_frame[column].mean())
            detail = (
                f"min={min_value}, max={max_value}, mean={mean_value}, unique={unique_count}"
            )
        else:
            top_values = (
                profile_frame[column]
                .astype("string")
                .value_counts(dropna=True)
                .head(max_categories)
                .index.tolist()
            )
            detail = f"unique={unique_count}, sample_values={top_values}"

        lines.append(f"- {column} ({dtype}): missing={missing}, {detail}")

    return "\n".join(lines)


def iter_dataframe_document_batches(
    df: pd.DataFrame,
    max_rows: int,
    batch_size: int = 10_000,
    max_columns: int = 40,
) -> Iterator[tuple[list[str], list[dict[str, Any]], list[str]]]:
    if df.empty:
        return

    capped_rows = min(max_rows, len(df))
    selected_columns = list(df.columns[:max_columns])
    truncated_columns = len(df.columns) - len(selected_columns)
    subset = df[selected_columns].head(capped_rows)

    docs: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []
    row_position = 0

    for source_index, row in subset.iterrows():
        row_parts = [f"{column}={_format_value(row[column])}" for column in selected_columns]
        if truncated_columns > 0:
            row_parts.append(f"omitted_columns={truncated_columns}")

        docs.append(" | ".join(row_parts))
        metadatas.append(
            {
                "row_position": int(row_position),
                "source_index": str(source_index),
            }
        )
        ids.append(f"row_{row_position}")
        row_position += 1

        if len(docs) >= batch_size:
            yield docs, metadatas, ids
            docs, metadatas, ids = [], [], []

    if docs:
        yield docs, metadatas, ids


def build_retrieval_preview(
    df: pd.DataFrame,
    metadatas: list[dict[str, Any]],
) -> pd.DataFrame:
    row_positions: list[int] = []
    for metadata in metadatas:
        position = metadata.get("row_position")
        if isinstance(position, int) and 0 <= position < len(df):
            row_positions.append(position)

    if not row_positions:
        return pd.DataFrame()

    unique_rows = list(dict.fromkeys(row_positions))
    return df.iloc[unique_rows].copy()


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "NULL"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _to_serializable(value: Any) -> str:
    if pd.isna(value):
        return "NULL"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _parse_likely_datetime_columns(df: pd.DataFrame) -> None:
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        column_name = str(column).strip().lower()
        if not any(token in column_name for token in ("date", "time", "timestamp")):
            continue

        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue

        converted = pd.to_datetime(series, errors="coerce")
        non_null_count = int(series.notna().sum())
        if non_null_count == 0:
            continue

        parse_ratio = float(converted.notna().sum() / non_null_count)
        if parse_ratio >= 0.85:
            df[column] = converted
