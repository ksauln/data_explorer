from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.compute_engine import dataframe_to_text, generate_sql_plan, run_sql_query
from src.answer_format_utils import format_answer_text
from src.cache_utils import cosine_similarity, normalize_question
from src.config import AppConfig
from src.data_utils import (
    build_retrieval_preview,
    dashboard_sample,
    dataset_overview,
    iter_dataframe_document_batches,
    load_dataframe,
    load_reference_text,
    normalize_table_name,
    schema_summary,
)
from src.llm_clients import create_llm_client
from src.logging_utils import append_jsonl_record, setup_logging, tail_log
from src.qa_engine import QAResponse, answer_with_context
from src.vector_store import (
    HashingEmbeddingFunction,
    OllamaEmbeddingFunction,
    OpenAIEmbeddingFunction,
    get_chroma_client,
    get_collection,
    index_document_batches,
    query_collection,
    rebuild_collection,
)

load_dotenv()
config = AppConfig.from_env()
logger = setup_logging(config.log_dir, log_level=logging.INFO)

st.set_page_config(
    page_title="Data Explorer QA",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_cached_chroma_client(chroma_dir: str):
    return get_chroma_client(Path(chroma_dir))


@st.cache_resource(show_spinner=False)
def get_cached_embedding_function(
    embedding_backend: str,
    openai_api_key: str,
    openai_embedding_model: str,
    ollama_base_url: str,
    ollama_embedding_model: str,
    hashing_embedding_dimensions: int,
):
    backend = embedding_backend.strip().lower()
    if backend == "openai":
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for OpenAI embeddings.")
        return OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model=openai_embedding_model,
        )
    if backend == "ollama":
        return OllamaEmbeddingFunction(
            base_url=ollama_base_url,
            model=ollama_embedding_model,
        )
    if backend == "hashing":
        return HashingEmbeddingFunction(n_features=int(hashing_embedding_dimensions))

    raise ValueError(f"Unsupported embedding backend: {embedding_backend}")


def _init_session_state() -> None:
    default_values = {
        "dataset_key": None,
        "df": None,
        "base_collection_name": None,
        "additional_tables_key": None,
        "additional_tables": {},
        "additional_tables_info": [],
        "table_context_key": None,
        "table_schema_summary_text": "",
        "table_catalog_text": "",
        "reference_key": None,
        "reference_text": "",
        "index_ready": False,
        "index_collection_name": "",
        "index_embedding_mode": "",
        "last_answer": "",
        "last_docs": [],
        "last_metadatas": [],
        "last_distances": [],
        "last_sql": "",
        "last_sql_rationale": "",
        "last_sql_error": "",
        "last_compute_df": pd.DataFrame(),
        "last_question": "",
        "last_reasoning_summary": "",
        "last_raw_llm_output": "",
        "last_qa_entry_id": "",
        "last_cache_status": "",
        "last_cache_similarity": 0.0,
        "qa_history": [],
        "qa_answer_cache": [],
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _build_collection_name(runtime_mode: str) -> str:
    base_name = st.session_state.base_collection_name or config.default_collection
    return f"{base_name}_{runtime_mode}"


def _build_embedding_mode(
    embedding_backend: str,
    openai_embedding_model: str,
    ollama_embedding_model: str,
    hashing_embedding_dimensions: int,
) -> str:
    if embedding_backend == "openai":
        return f"openai:{openai_embedding_model}"
    if embedding_backend == "ollama":
        return f"ollama:{ollama_embedding_model}"
    return f"hashing:{int(hashing_embedding_dimensions)}"


def _load_uploaded_dataset(file_name: str, file_bytes: bytes) -> None:
    dataset_key = hashlib.sha1(file_bytes).hexdigest()[:12]
    if st.session_state.dataset_key == dataset_key:
        return

    logger.info("Loading new dataset %s", file_name)
    df = load_dataframe(file_name, file_bytes)
    st.session_state.dataset_key = dataset_key
    st.session_state.df = df
    st.session_state.base_collection_name = f"{config.default_collection}_{dataset_key}"
    st.session_state.table_context_key = None
    st.session_state.table_schema_summary_text = ""
    st.session_state.table_catalog_text = ""
    st.session_state.index_ready = False
    st.session_state.index_collection_name = ""
    st.session_state.index_embedding_mode = ""
    st.session_state.last_answer = ""
    st.session_state.last_docs = []
    st.session_state.last_metadatas = []
    st.session_state.last_distances = []
    st.session_state.last_sql = ""
    st.session_state.last_sql_rationale = ""
    st.session_state.last_sql_error = ""
    st.session_state.last_compute_df = pd.DataFrame()
    st.session_state.last_question = ""
    st.session_state.last_reasoning_summary = ""
    st.session_state.last_raw_llm_output = ""
    st.session_state.last_qa_entry_id = ""
    st.session_state.last_cache_status = ""
    st.session_state.last_cache_similarity = 0.0
    st.session_state.qa_history = []
    st.session_state.qa_answer_cache = []
    logger.info("Dataset loaded successfully with shape %s", df.shape)


def _load_additional_tables(uploaded_files: list[st.runtime.uploaded_file_manager.UploadedFile]) -> None:
    if not uploaded_files:
        if st.session_state.additional_tables:
            logger.info("Clearing additional SQL join tables")
        st.session_state.additional_tables_key = None
        st.session_state.additional_tables = {}
        st.session_state.additional_tables_info = []
        st.session_state.table_context_key = None
        st.session_state.table_schema_summary_text = ""
        st.session_state.table_catalog_text = ""
        return

    signatures: list[str] = []
    tables: dict[str, pd.DataFrame] = {}
    table_info: list[dict[str, str | int]] = []
    used_names: set[str] = {"data"}

    for file in uploaded_files:
        file_bytes = file.getvalue()
        digest = hashlib.sha1(file_bytes).hexdigest()[:12]
        signatures.append(f"{file.name}:{digest}")

        base_name = normalize_table_name(file.name)
        table_name = base_name
        suffix = 2
        while table_name in used_names:
            table_name = f"{base_name}_{suffix}"
            suffix += 1

        table_df = load_dataframe(file.name, file_bytes)
        tables[table_name] = table_df
        used_names.add(table_name)
        table_info.append(
            {
                "table_name": table_name,
                "source_file": file.name,
                "rows": int(table_df.shape[0]),
                "columns": int(table_df.shape[1]),
            }
        )

    new_signature = "|".join(sorted(signatures))
    if st.session_state.additional_tables_key == new_signature:
        return

    logger.info("Loaded %s additional SQL tables", len(tables))
    st.session_state.additional_tables_key = new_signature
    st.session_state.additional_tables = tables
    st.session_state.additional_tables_info = table_info
    st.session_state.table_context_key = None
    st.session_state.table_schema_summary_text = ""
    st.session_state.table_catalog_text = ""


def _load_reference_context(file_name: str, file_bytes: bytes) -> None:
    reference_key = hashlib.sha1(file_bytes).hexdigest()[:12]
    if st.session_state.reference_key == reference_key:
        return

    logger.info("Loading reference context %s", file_name)
    reference_text = load_reference_text(file_name, file_bytes)
    st.session_state.reference_key = reference_key
    st.session_state.reference_text = reference_text
    logger.info("Reference context loaded (%s characters)", len(reference_text))


def _get_table_context(
    primary_df: pd.DataFrame,
    additional_tables: dict[str, pd.DataFrame],
) -> tuple[str, str]:
    context_key = (
        f"{st.session_state.dataset_key}:{st.session_state.additional_tables_key}"
    )
    if st.session_state.table_context_key == context_key:
        return (
            st.session_state.table_schema_summary_text,
            st.session_state.table_catalog_text,
        )

    schema_sections = [
        "Table: data (primary)\n" + schema_summary(primary_df, max_profile_rows=200_000)
    ]
    catalog_lines = [
        f"- data (primary): rows={len(primary_df)}, columns={primary_df.shape[1]}"
    ]

    for table_name, table_df in additional_tables.items():
        schema_sections.append(
            f"Table: {table_name}\n" + schema_summary(table_df, max_profile_rows=100_000)
        )
        catalog_lines.append(
            f"- {table_name}: rows={len(table_df)}, columns={table_df.shape[1]}"
        )

    st.session_state.table_schema_summary_text = "\n\n".join(schema_sections)
    st.session_state.table_catalog_text = "\n".join(catalog_lines)
    st.session_state.table_context_key = context_key
    return st.session_state.table_schema_summary_text, st.session_state.table_catalog_text


def _append_qa_history_entry(
    *,
    question: str,
    runtime_mode: str,
    embedding_mode: str,
    enable_compute: bool,
    enable_rag: bool,
    status: str,
    answer: str = "",
    reasoning_summary: str = "",
    generated_sql: str = "",
    sql_rationale: str = "",
    sql_error: str = "",
    compute_rows: int = 0,
    retrieved_count: int = 0,
    error_message: str = "",
) -> str:
    entry_id = uuid4().hex
    history_entry = {
        "entry_id": entry_id,
        "asked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question.strip(),
        "status": status,
        "runtime_mode": runtime_mode,
        "embedding_mode": embedding_mode,
        "compute_enabled": bool(enable_compute),
        "rag_enabled": bool(enable_rag),
        "retrieved_docs": int(retrieved_count),
        "compute_rows": int(compute_rows),
        "answer": answer.strip(),
        "reasoning_summary": reasoning_summary.strip(),
        "sql": generated_sql.strip(),
        "sql_rationale": sql_rationale.strip(),
        "sql_error": sql_error.strip(),
        "error_message": error_message.strip(),
        "feedback": "",
        "feedback_recorded_at": "",
    }
    history: list[dict[str, object]] = list(st.session_state.qa_history)
    history.append(history_entry)
    st.session_state.qa_history = history[-100:]
    logger.info(
        "Q&A history updated: status=%s, entries=%s",
        status,
        len(st.session_state.qa_history),
    )
    return entry_id


def _set_feedback_for_entry(entry_id: str, feedback_value: str) -> None:
    if not entry_id:
        return

    normalized = feedback_value.strip().lower()
    if normalized not in {"up", "down"}:
        return

    recorded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    updated = False
    feedback_record: dict[str, object] = {
        "recorded_at": recorded_at,
        "entry_id": entry_id,
        "feedback": normalized,
    }

    history: list[dict[str, object]] = list(st.session_state.qa_history)
    for entry in history:
        if str(entry.get("entry_id", "")) != entry_id:
            continue
        entry["feedback"] = normalized
        entry["feedback_recorded_at"] = recorded_at
        updated = True
        feedback_record.update(
            {
                "asked_at": entry.get("asked_at"),
                "question": entry.get("question"),
                "status": entry.get("status"),
                "runtime_mode": entry.get("runtime_mode"),
                "embedding_mode": entry.get("embedding_mode"),
                "compute_rows": entry.get("compute_rows"),
                "retrieved_docs": entry.get("retrieved_docs"),
            }
        )
        break

    if not updated:
        return

    st.session_state.qa_history = history
    try:
        append_jsonl_record(config.feedback_log_file, feedback_record)
        logger.info("Feedback recorded for entry_id=%s feedback=%s", entry_id, normalized)
    except Exception as exc:
        logger.exception("Failed to write feedback record: %s", exc)


def _render_feedback_controls(entry_id: str) -> None:
    if not entry_id:
        return

    feedback_value = ""
    for entry in st.session_state.qa_history:
        if str(entry.get("entry_id", "")) == entry_id:
            feedback_value = str(entry.get("feedback", "")).strip().lower()
            break

    st.caption("Rate this answer")
    col_up, col_down, col_state = st.columns([1, 1, 4])
    with col_up:
        if st.button("👍", key=f"feedback_up_{entry_id}"):
            _set_feedback_for_entry(entry_id, "up")
            st.rerun()
    with col_down:
        if st.button("👎", key=f"feedback_down_{entry_id}"):
            _set_feedback_for_entry(entry_id, "down")
            st.rerun()
    with col_state:
        if feedback_value == "up":
            st.success("Feedback: thumbs up")
        elif feedback_value == "down":
            st.error("Feedback: thumbs down")
        else:
            st.info("No feedback submitted yet.")


def _build_cache_context_signature(
    *,
    runtime_mode: str,
    embedding_mode: str,
    enable_compute: bool,
    enable_rag: bool,
    top_k: int,
    index_compatible: bool,
) -> str:
    return "|".join(
        [
            f"dataset={st.session_state.dataset_key}",
            f"joins={st.session_state.additional_tables_key or '-'}",
            f"dict={st.session_state.reference_key or '-'}",
            f"runtime={runtime_mode}",
            f"embed={embedding_mode}",
            f"compute={int(enable_compute)}",
            f"rag={int(enable_rag)}",
            f"top_k={int(top_k)}",
            f"index_ready={int(index_compatible)}",
        ]
    )


def _extract_embedding_vector(
    embedding_function: object | None,
    question_text: str,
) -> list[float] | None:
    if embedding_function is None:
        return None

    try:
        if hasattr(embedding_function, "embed_query"):
            result = embedding_function.embed_query(question_text)
        else:
            result = embedding_function([question_text])
    except Exception as exc:
        logger.warning("Question embedding failed for cache lookup: %s", exc)
        return None

    if not isinstance(result, list) or not result:
        return None

    first = result[0]
    if hasattr(first, "tolist"):
        first = first.tolist()
    if not isinstance(first, list) or not first:
        return None

    try:
        return [float(value) for value in first]
    except Exception:
        return None


def _find_cached_answer(
    *,
    question_text: str,
    context_signature: str,
    semantic_enabled: bool,
    semantic_threshold: float,
    embedding_function: object | None,
) -> tuple[dict[str, object] | None, str, float, list[float] | None]:
    cache_entries = [
        entry
        for entry in st.session_state.qa_answer_cache
        if str(entry.get("context_signature", "")) == context_signature
    ]
    if not cache_entries:
        return None, "", 0.0, None

    normalized_question = normalize_question(question_text)
    for entry in reversed(cache_entries):
        if str(entry.get("normalized_question", "")) == normalized_question:
            return entry, "exact", 1.0, None

    if not semantic_enabled:
        return None, "", 0.0, None

    question_embedding = _extract_embedding_vector(embedding_function, question_text)
    if not question_embedding:
        return None, "", 0.0, None

    best_entry: dict[str, object] | None = None
    best_score = -1.0
    for entry in cache_entries:
        candidate_embedding = entry.get("question_embedding")
        if not isinstance(candidate_embedding, list) or not candidate_embedding:
            continue
        score = cosine_similarity(question_embedding, candidate_embedding)
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry is None or best_score < semantic_threshold:
        return None, "", 0.0, question_embedding

    return best_entry, "semantic", float(best_score), question_embedding


def _hydrate_last_answer_from_cache(cache_entry: dict[str, object]) -> None:
    st.session_state.last_question = str(cache_entry.get("question", ""))
    cached_answer = str(cache_entry.get("answer", ""))
    st.session_state.last_answer = format_answer_text(
        cached_answer,
        question=st.session_state.last_question,
        computed_df=cache_entry.get("computed_df")
        if isinstance(cache_entry.get("computed_df"), pd.DataFrame)
        else pd.DataFrame(),
    )
    st.session_state.last_reasoning_summary = str(cache_entry.get("reasoning_summary", ""))
    st.session_state.last_raw_llm_output = str(cache_entry.get("raw_model_output", ""))
    st.session_state.last_sql = str(cache_entry.get("generated_sql", ""))
    st.session_state.last_sql_rationale = str(cache_entry.get("sql_rationale", ""))
    st.session_state.last_sql_error = str(cache_entry.get("sql_error", ""))
    st.session_state.last_docs = list(cache_entry.get("retrieved_docs", []))
    st.session_state.last_metadatas = list(cache_entry.get("retrieved_metadatas", []))
    st.session_state.last_distances = list(cache_entry.get("retrieved_distances", []))

    cached_df = cache_entry.get("computed_df")
    if isinstance(cached_df, pd.DataFrame):
        st.session_state.last_compute_df = cached_df.copy()
    else:
        st.session_state.last_compute_df = pd.DataFrame()


def _store_answer_cache_entry(
    *,
    question_text: str,
    context_signature: str,
    answer: str,
    reasoning_summary: str,
    raw_model_output: str,
    generated_sql: str,
    sql_rationale: str,
    sql_error: str,
    computed_df: pd.DataFrame,
    retrieved_docs: list[str],
    retrieved_metadatas: list[dict[str, object]],
    retrieved_distances: list[float],
    question_embedding: list[float] | None,
    max_entries: int,
) -> None:
    cache_entry = {
        "cache_id": uuid4().hex,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question_text.strip(),
        "normalized_question": normalize_question(question_text),
        "context_signature": context_signature,
        "answer": answer.strip(),
        "reasoning_summary": reasoning_summary.strip(),
        "raw_model_output": raw_model_output,
        "generated_sql": generated_sql.strip(),
        "sql_rationale": sql_rationale.strip(),
        "sql_error": sql_error.strip(),
        "computed_df": computed_df.copy() if isinstance(computed_df, pd.DataFrame) else pd.DataFrame(),
        "retrieved_docs": list(retrieved_docs),
        "retrieved_metadatas": list(retrieved_metadatas),
        "retrieved_distances": list(retrieved_distances),
        "question_embedding": list(question_embedding) if question_embedding else [],
    }

    cache_entries: list[dict[str, object]] = list(st.session_state.qa_answer_cache)
    cache_entries.append(cache_entry)
    st.session_state.qa_answer_cache = cache_entries[-max(1, int(max_entries)) :]
    logger.info(
        "Stored answer in cache (entries=%s, question=%s)",
        len(st.session_state.qa_answer_cache),
        question_text,
    )


def _render_qa_history() -> None:
    st.subheader("Q&A Session History")
    history: list[dict[str, object]] = list(st.session_state.qa_history)
    if not history:
        st.info("No questions asked in this session yet.")
        return

    col_left, col_right = st.columns([1, 4])
    with col_left:
        if st.button("Clear History"):
            st.session_state.qa_history = []
            st.rerun()
    with col_right:
        st.caption(f"Session entries: {len(history)} (latest 100 retained)")

    history_df = pd.DataFrame(history[::-1])
    if "answer" not in history_df.columns:
        history_df["answer"] = ""
    history_df["answer_preview"] = history_df["answer"].astype(str).str.slice(0, 160)
    display_columns = [
        "asked_at",
        "status",
        "feedback",
        "question",
        "runtime_mode",
        "embedding_mode",
        "compute_rows",
        "retrieved_docs",
        "answer_preview",
    ]
    for column in display_columns:
        if column not in history_df.columns:
            history_df[column] = ""
    st.dataframe(
        history_df[display_columns],
        width="stretch",
    )

    for entry in history[::-1][:10]:
        question_text = str(entry.get("question", ""))
        label = f"{entry.get('asked_at', '')} | {entry.get('status', '')} | {question_text[:90]}"
        with st.expander(label):
            st.markdown("**Question**")
            st.write(question_text)
            st.markdown("**Answer**")
            answer_text = str(entry.get("answer", "")).strip() or "No answer captured."
            st.write(answer_text)

            reasoning_summary = str(entry.get("reasoning_summary", "")).strip()
            if reasoning_summary:
                st.markdown("**LLM Reasoning Summary**")
                st.write(reasoning_summary)

            st.markdown("**Details**")
            st.write(
                {
                    "runtime_mode": entry.get("runtime_mode"),
                    "embedding_mode": entry.get("embedding_mode"),
                    "compute_rows": entry.get("compute_rows"),
                    "retrieved_docs": entry.get("retrieved_docs"),
                    "feedback": entry.get("feedback") or "none",
                }
            )

            sql_text = str(entry.get("sql", "")).strip()
            if sql_text:
                st.markdown("**Generated SQL**")
                st.code(sql_text, language="sql")

            sql_error = str(entry.get("sql_error", "")).strip()
            if sql_error:
                st.markdown("**SQL Error**")
                st.code(sql_error, language="text")

            error_message = str(entry.get("error_message", "")).strip()
            if error_message:
                st.markdown("**Error Message**")
                st.code(error_message, language="text")


def _render_dashboard(df: pd.DataFrame) -> None:
    st.subheader("Dashboard")
    viz_df = dashboard_sample(df, max_rows=200_000)
    if len(viz_df) < len(df):
        st.caption(
            f"Dashboard charts use a sampled subset of {len(viz_df):,} rows for performance."
        )

    missing_df = (
        viz_df.isna().sum().reset_index(name="missing_count").rename(columns={"index": "column"})
    )
    missing_df = missing_df[missing_df["missing_count"] > 0]

    if not missing_df.empty:
        fig_missing = px.bar(
            missing_df,
            x="column",
            y="missing_count",
            title="Missing Values by Column",
        )
        st.plotly_chart(fig_missing, width="stretch")
    else:
        st.info("No missing values detected in sampled dashboard data.")

    numeric_columns = viz_df.select_dtypes(include="number").columns.tolist()
    categorical_columns = viz_df.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    col_left, col_right = st.columns(2)
    with col_left:
        if numeric_columns:
            selected_numeric = st.selectbox(
                "Distribution Column",
                options=numeric_columns,
                key="dist_column",
            )
            fig_hist = px.histogram(
                viz_df,
                x=selected_numeric,
                nbins=40,
                title=f"Distribution: {selected_numeric}",
            )
            st.plotly_chart(fig_hist, width="stretch")
        else:
            st.info("No numeric columns available for distribution plots.")

    with col_right:
        if categorical_columns:
            selected_category = st.selectbox(
                "Top Values Column",
                options=categorical_columns,
                key="top_values_column",
            )
            top_values = (
                viz_df[selected_category]
                .astype("string")
                .fillna("NULL")
                .value_counts()
                .head(20)
                .reset_index()
            )
            top_values.columns = [selected_category, "count"]
            fig_bar = px.bar(
                top_values,
                x=selected_category,
                y="count",
                title=f"Top Values: {selected_category}",
            )
            st.plotly_chart(fig_bar, width="stretch")
        else:
            st.info("No categorical columns available for top-value charts.")

    if len(numeric_columns) >= 2:
        x_col, y_col = st.columns(2)
        with x_col:
            scatter_x = st.selectbox("Scatter X", numeric_columns, key="scatter_x")
        with y_col:
            scatter_y = st.selectbox("Scatter Y", numeric_columns, index=1, key="scatter_y")

        scatter_frame = viz_df[[scatter_x, scatter_y]].dropna().head(10_000)
        if not scatter_frame.empty:
            fig_scatter = px.scatter(
                scatter_frame,
                x=scatter_x,
                y=scatter_y,
                title=f"Scatter: {scatter_x} vs {scatter_y}",
            )
            st.plotly_chart(fig_scatter, width="stretch")

    if len(numeric_columns) >= 2:
        corr_columns = numeric_columns[:20]
        corr_matrix = viz_df[corr_columns].corr(numeric_only=True)
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Correlation Heatmap",
            aspect="auto",
        )
        st.plotly_chart(fig_corr, width="stretch")


def _question_requests_visual(question: str) -> bool:
    normalized = normalize_question(question)
    if not normalized:
        return False
    keywords = (
        "chart",
        "graph",
        "plot",
        "visual",
        "trend",
        "line",
        "bar",
        "timeline",
    )
    return any(keyword in normalized for keyword in keywords)


def _maybe_coerce_datetime(series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if not (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        return None

    converted = pd.to_datetime(series, errors="coerce")
    non_null = int(series.notna().sum())
    if non_null == 0:
        return None
    ratio = float(converted.notna().sum() / non_null)
    return converted if ratio >= 0.7 else None


def _infer_qa_chart(
    computed_df: pd.DataFrame,
    question: str,
) -> tuple[object | None, str]:
    if computed_df.empty:
        return None, "No computed data available."

    plot_df = computed_df.copy()
    numeric_columns = [
        column
        for column in plot_df.select_dtypes(include="number").columns
        if str(plot_df[column].dtype) != "bool"
    ]
    if not numeric_columns:
        return None, "No numeric measure found in query results."

    datetime_candidates: list[str] = []
    for column in plot_df.columns:
        lowered = str(column).lower()
        series = plot_df[column]
        coerced = _maybe_coerce_datetime(series)
        if coerced is not None:
            plot_df[column] = coerced
            if any(token in lowered for token in ("date", "time", "month", "year")):
                datetime_candidates.insert(0, column)
            else:
                datetime_candidates.append(column)

    color_column = ""
    if datetime_candidates:
        x_column = datetime_candidates[0]
        y_column = numeric_columns[0]
        chart_type = "line"
        for column in plot_df.columns:
            if column in {x_column, y_column}:
                continue
            if pd.api.types.is_object_dtype(plot_df[column]) or pd.api.types.is_string_dtype(
                plot_df[column]
            ):
                unique_count = int(plot_df[column].nunique(dropna=True))
                if 2 <= unique_count <= 12:
                    color_column = column
                    break
        if color_column:
            top_categories = (
                plot_df.groupby(color_column, dropna=False)[y_column]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .index
            )
            plot_df = plot_df[plot_df[color_column].isin(top_categories)]
    else:
        y_column = numeric_columns[0]
        x_column = ""
        for column in plot_df.columns:
            if column == y_column:
                continue
            if pd.api.types.is_object_dtype(plot_df[column]) or pd.api.types.is_string_dtype(
                plot_df[column]
            ):
                x_column = column
                break
        if not x_column:
            x_column = str(plot_df.index.name or "row_index")
            plot_df = plot_df.reset_index(names=x_column)
        chart_type = "bar"

    normalized_question = normalize_question(question)
    if "bar" in normalized_question:
        chart_type = "bar"
    elif "line" in normalized_question or "trend" in normalized_question:
        chart_type = "line"

    title = f"{y_column} by {x_column}"
    if chart_type == "line":
        fig = px.line(
            plot_df,
            x=x_column,
            y=y_column,
            color=color_column or None,
            markers=True,
            title=title,
        )
    else:
        fig = px.bar(
            plot_df,
            x=x_column,
            y=y_column,
            color=color_column or None,
            title=title,
        )
    return fig, ""


def _render_qa(
    df: pd.DataFrame,
    *,
    additional_tables: dict[str, pd.DataFrame],
    additional_tables_info: list[dict[str, str | int]],
    runtime_mode: str,
    embedding_backend: str,
    openai_api_key: str | None,
    openai_model: str,
    openai_embedding_model: str,
    ollama_base_url: str,
    ollama_model: str,
    ollama_embedding_model: str,
    hashing_embedding_dimensions: int,
    rows_to_index: int,
    index_batch_size: int,
    top_k: int,
    enable_compute: bool,
    enable_rag: bool,
    enable_answer_cache: bool,
    enable_semantic_cache: bool,
    semantic_cache_threshold: float,
    answer_cache_max_entries: int,
) -> None:
    st.subheader("Natural Language Q&A")
    collection_name = _build_collection_name(runtime_mode)
    embedding_mode = _build_embedding_mode(
        embedding_backend,
        openai_embedding_model,
        ollama_embedding_model,
        hashing_embedding_dimensions,
    )
    provider = "openai" if runtime_mode == "cloud" else "ollama"

    index_compatible = (
        st.session_state.index_ready
        and st.session_state.index_collection_name == collection_name
        and st.session_state.index_embedding_mode == embedding_mode
    )
    st.caption(
        f"Runtime: `{runtime_mode}` | LLM: `{provider}` | Embeddings: `{embedding_mode}` | "
        f"Collection: `{collection_name}`"
    )
    if additional_tables_info:
        join_table_names = ", ".join(
            str(item.get("table_name")) for item in additional_tables_info
        )
        st.caption(f"Additional SQL tables: `{join_table_names}`")
    if enable_answer_cache:
        semantic_mode = (
            f"semantic on (threshold={semantic_cache_threshold:.2f})"
            if enable_semantic_cache
            else "semantic off"
        )
        st.caption(f"Answer cache: enabled, {semantic_mode}, max_entries={answer_cache_max_entries}")
    else:
        st.caption("Answer cache: disabled")

    if not enable_compute and not enable_rag:
        st.warning("Enable at least one answer path (Compute or RAG).")
        return

    chroma_client = get_cached_chroma_client(str(config.chroma_dir))

    if enable_rag and st.button("Build / Refresh QA Index", type="primary"):
        try:
            embedding_function = get_cached_embedding_function(
                embedding_backend=embedding_backend,
                openai_api_key=openai_api_key or "",
                openai_embedding_model=openai_embedding_model,
                ollama_base_url=ollama_base_url,
                ollama_embedding_model=ollama_embedding_model,
                hashing_embedding_dimensions=hashing_embedding_dimensions,
            )
        except Exception as exc:
            st.error(f"Embedding setup failed: {exc}")
            logger.exception("Embedding setup failed during indexing: %s", exc)
            return

        total_rows = min(rows_to_index, len(df))
        logger.info(
            "Starting index build for %s (%s rows, batch_size=%s)",
            collection_name,
            total_rows,
            index_batch_size,
        )
        with st.spinner("Indexing rows in ChromaDB..."):
            collection = rebuild_collection(
                chroma_client,
                collection_name=collection_name,
                embedding_function=embedding_function,
            )
            progress = st.progress(0, text=f"Indexed 0 / {total_rows:,} rows")

            def on_batch_indexed(indexed_count: int) -> None:
                fraction = 1.0 if total_rows == 0 else min(indexed_count / total_rows, 1.0)
                progress.progress(
                    int(fraction * 100),
                    text=f"Indexed {indexed_count:,} / {total_rows:,} rows",
                )
                if indexed_count % 100_000 == 0:
                    logger.info("Indexing progress: %s/%s", indexed_count, total_rows)

            batches = iter_dataframe_document_batches(
                df,
                max_rows=rows_to_index,
                batch_size=index_batch_size,
            )
            indexed_count = index_document_batches(
                collection,
                batches=batches,
                on_batch_indexed=on_batch_indexed,
            )
            progress.progress(100, text=f"Indexed {indexed_count:,} / {total_rows:,} rows")

        st.session_state.index_ready = True
        st.session_state.index_collection_name = collection_name
        st.session_state.index_embedding_mode = embedding_mode
        logger.info("Index build complete for %s; indexed_count=%s", collection_name, indexed_count)
        st.success(f"Indexed {indexed_count:,} rows into ChromaDB.")

    question = st.text_area(
        "Ask a question about this dataset",
        placeholder="Example: Which segment has the highest average revenue?",
        height=100,
    )

    if st.button("Ask Question"):
        question_text = question.strip()
        if not question_text:
            st.warning("Please enter a question.")
            return

        st.session_state.last_answer = ""
        st.session_state.last_docs = []
        st.session_state.last_metadatas = []
        st.session_state.last_distances = []
        st.session_state.last_sql = ""
        st.session_state.last_sql_rationale = ""
        st.session_state.last_sql_error = ""
        st.session_state.last_compute_df = pd.DataFrame()
        st.session_state.last_question = question_text
        st.session_state.last_reasoning_summary = ""
        st.session_state.last_raw_llm_output = ""
        st.session_state.last_qa_entry_id = ""
        st.session_state.last_cache_status = ""
        st.session_state.last_cache_similarity = 0.0

        cache_context_signature = _build_cache_context_signature(
            runtime_mode=runtime_mode,
            embedding_mode=embedding_mode,
            enable_compute=enable_compute,
            enable_rag=enable_rag,
            top_k=top_k,
            index_compatible=index_compatible,
        )

        cache_embedding_function: object | None = None
        if enable_answer_cache and enable_semantic_cache:
            try:
                cache_embedding_function = get_cached_embedding_function(
                    embedding_backend=embedding_backend,
                    openai_api_key=openai_api_key or "",
                    openai_embedding_model=openai_embedding_model,
                    ollama_base_url=ollama_base_url,
                    ollama_embedding_model=ollama_embedding_model,
                    hashing_embedding_dimensions=hashing_embedding_dimensions,
                )
            except Exception as exc:
                logger.warning("Semantic cache disabled for this query due to embedding setup error: %s", exc)
                cache_embedding_function = None

        cached_entry: dict[str, object] | None = None
        cache_mode = ""
        cache_similarity = 0.0
        question_embedding: list[float] | None = None
        if enable_answer_cache:
            cached_entry, cache_mode, cache_similarity, question_embedding = _find_cached_answer(
                question_text=question_text,
                context_signature=cache_context_signature,
                semantic_enabled=enable_semantic_cache,
                semantic_threshold=float(semantic_cache_threshold),
                embedding_function=cache_embedding_function,
            )
            if cached_entry is not None:
                _hydrate_last_answer_from_cache(cached_entry)
                st.session_state.last_question = question_text
                st.session_state.last_cache_status = cache_mode
                st.session_state.last_cache_similarity = float(cache_similarity)
                logger.info(
                    "Answer cache hit: mode=%s similarity=%.3f question=%s",
                    cache_mode,
                    cache_similarity,
                    question_text,
                )
                entry_id = _append_qa_history_entry(
                    question=question_text,
                    runtime_mode=runtime_mode,
                    embedding_mode=embedding_mode,
                    enable_compute=enable_compute,
                    enable_rag=enable_rag,
                    status=f"cache_hit_{cache_mode}",
                    answer=st.session_state.last_answer,
                    reasoning_summary=st.session_state.last_reasoning_summary,
                    generated_sql=st.session_state.last_sql,
                    sql_rationale=st.session_state.last_sql_rationale,
                    sql_error=st.session_state.last_sql_error,
                    compute_rows=len(st.session_state.last_compute_df),
                    retrieved_count=len(st.session_state.last_docs),
                    error_message=(
                        f"Served from answer cache ({cache_mode}, similarity={cache_similarity:.3f})."
                    ),
                )
                st.session_state.last_qa_entry_id = entry_id
                st.rerun()
                return

        query_progress = st.progress(0, text="Preparing query...")

        def update_query_progress(value: int, text: str) -> None:
            query_progress.progress(value, text=text)
            logger.info("Query progress: %s%% - %s", value, text)

        retrieved_docs: list[str] = []
        retrieved_metadatas: list[dict[str, object]] = []
        retrieved_distances: list[float] = []
        generated_sql = ""
        sql_rationale = ""
        sql_error = ""
        computed_df = pd.DataFrame()
        computed_result_text = ""

        update_query_progress(10, "Profiling table schemas...")
        schema_text, table_catalog_text = _get_table_context(df, additional_tables)
        llm_client = None
        try:
            update_query_progress(20, "Initializing language model client...")
            llm_client = create_llm_client(
                provider=provider,
                openai_api_key=openai_api_key,
                openai_model=openai_model,
                ollama_base_url=ollama_base_url,
                ollama_model=ollama_model,
            )
            update_query_progress(30, "Model client ready.")
        except Exception as exc:
            update_query_progress(100, "Failed to initialize model client.")
            st.error(f"LLM configuration failed: {exc}")
            logger.exception("LLM client setup failed: %s", exc)
            entry_id = _append_qa_history_entry(
                question=question_text,
                runtime_mode=runtime_mode,
                embedding_mode=embedding_mode,
                enable_compute=enable_compute,
                enable_rag=enable_rag,
                status="failed_llm_init",
                error_message=str(exc),
            )
            st.session_state.last_qa_entry_id = entry_id
            return

        if enable_compute:
            try:
                update_query_progress(40, "Running compute path (SQL planning + execution)...")
                logger.info("Generating SQL plan for question: %s", question_text)
                sql_plan = generate_sql_plan(
                    question=question_text,
                    schema_summary_text=schema_text,
                    dictionary_text=st.session_state.reference_text,
                    table_catalog_text=table_catalog_text,
                    default_table_name="data",
                    llm_client=llm_client,
                )
                generated_sql = sql_plan.sql
                sql_rationale = sql_plan.rationale
                computed_df = run_sql_query(
                    df,
                    generated_sql,
                    additional_tables=additional_tables,
                    primary_table_name="data",
                )
                computed_result_text = dataframe_to_text(computed_df, max_rows=40)
                logger.info(
                    "Compute query executed successfully; returned_rows=%s",
                    len(computed_df),
                )
                update_query_progress(60, "Compute path complete.")
            except Exception as exc:
                sql_error = str(exc)
                logger.exception("Compute path failed: %s", exc)
                update_query_progress(60, "Compute path failed, continuing with other context.")
        else:
            update_query_progress(60, "Compute path skipped.")

        if enable_rag:
            if not index_compatible:
                st.info(
                    "RAG index is not built for the current runtime/embedding settings. "
                    "Build the index to enable retrieval context."
                )
                update_query_progress(80, "Retrieval skipped (index not ready).")
            else:
                try:
                    update_query_progress(70, "Running retrieval from ChromaDB...")
                    embedding_function = get_cached_embedding_function(
                        embedding_backend=embedding_backend,
                        openai_api_key=openai_api_key or "",
                        openai_embedding_model=openai_embedding_model,
                        ollama_base_url=ollama_base_url,
                        ollama_embedding_model=ollama_embedding_model,
                        hashing_embedding_dimensions=hashing_embedding_dimensions,
                    )
                    collection = get_collection(
                        chroma_client,
                        collection_name=collection_name,
                        embedding_function=embedding_function,
                    )
                    retrieved_docs, retrieved_metadatas, retrieved_distances = query_collection(
                        collection,
                        question_text,
                        top_k=top_k,
                    )
                    logger.info("Retrieved %s documents for question", len(retrieved_docs))
                    update_query_progress(80, "Retrieval complete.")
                except Exception as exc:
                    logger.exception("RAG retrieval failed: %s", exc)
                    st.error(f"RAG retrieval failed: {exc}")
                    update_query_progress(80, "Retrieval failed, continuing with available context.")
        else:
            update_query_progress(80, "Retrieval path skipped.")

        if not computed_result_text and not retrieved_docs:
            update_query_progress(100, "No context found for this question.")
            st.warning(
                "No usable context found. Try enabling compute, building the RAG index, or asking a narrower question."
            )
            st.session_state.last_sql = generated_sql
            st.session_state.last_sql_rationale = sql_rationale
            st.session_state.last_sql_error = sql_error
            st.session_state.last_compute_df = computed_df
            entry_id = _append_qa_history_entry(
                question=question_text,
                runtime_mode=runtime_mode,
                embedding_mode=embedding_mode,
                enable_compute=enable_compute,
                enable_rag=enable_rag,
                status="no_context",
                generated_sql=generated_sql,
                sql_rationale=sql_rationale,
                sql_error=sql_error,
                compute_rows=len(computed_df),
                retrieved_count=len(retrieved_docs),
                error_message="No usable compute or retrieval context.",
            )
            st.session_state.last_qa_entry_id = entry_id
            return

        try:
            update_query_progress(90, "Synthesizing final answer...")
            qa_response: QAResponse = answer_with_context(
                question=question_text,
                overview=dataset_overview(df),
                schema_summary_text=schema_text,
                dictionary_text=st.session_state.reference_text,
                retrieved_documents=retrieved_docs,
                computed_result_text=computed_result_text,
                generated_sql=generated_sql,
                llm_client=llm_client,
            )
            formatted_answer = format_answer_text(
                qa_response.answer,
                question=question_text,
                computed_df=computed_df,
            )
            st.session_state.last_answer = formatted_answer
            st.session_state.last_docs = retrieved_docs
            st.session_state.last_metadatas = retrieved_metadatas
            st.session_state.last_distances = retrieved_distances
            st.session_state.last_sql = generated_sql
            st.session_state.last_sql_rationale = sql_rationale
            st.session_state.last_sql_error = sql_error
            st.session_state.last_compute_df = computed_df
            st.session_state.last_reasoning_summary = qa_response.reasoning_summary
            st.session_state.last_raw_llm_output = qa_response.raw_model_output
            logger.info("Question answered successfully.")
            entry_id = _append_qa_history_entry(
                question=question_text,
                runtime_mode=runtime_mode,
                embedding_mode=embedding_mode,
                enable_compute=enable_compute,
                enable_rag=enable_rag,
                status="completed",
                answer=formatted_answer,
                reasoning_summary=qa_response.reasoning_summary,
                generated_sql=generated_sql,
                sql_rationale=sql_rationale,
                sql_error=sql_error,
                compute_rows=len(computed_df),
                retrieved_count=len(retrieved_docs),
            )
            st.session_state.last_qa_entry_id = entry_id
            if enable_answer_cache:
                if question_embedding is None and enable_semantic_cache:
                    question_embedding = _extract_embedding_vector(cache_embedding_function, question_text)
                _store_answer_cache_entry(
                    question_text=question_text,
                    context_signature=cache_context_signature,
                    answer=formatted_answer,
                    reasoning_summary=qa_response.reasoning_summary,
                    raw_model_output=qa_response.raw_model_output,
                    generated_sql=generated_sql,
                    sql_rationale=sql_rationale,
                    sql_error=sql_error,
                    computed_df=computed_df,
                    retrieved_docs=retrieved_docs,
                    retrieved_metadatas=retrieved_metadatas,
                    retrieved_distances=retrieved_distances,
                    question_embedding=question_embedding,
                    max_entries=answer_cache_max_entries,
                )
            update_query_progress(100, "Answer ready.")
        except Exception as exc:
            logger.exception("Answer synthesis failed: %s", exc)
            st.error(f"Failed to produce final answer: {exc}")
            entry_id = _append_qa_history_entry(
                question=question_text,
                runtime_mode=runtime_mode,
                embedding_mode=embedding_mode,
                enable_compute=enable_compute,
                enable_rag=enable_rag,
                status="failed_answer_synthesis",
                generated_sql=generated_sql,
                sql_rationale=sql_rationale,
                sql_error=sql_error,
                compute_rows=len(computed_df),
                retrieved_count=len(retrieved_docs),
                error_message=str(exc),
            )
            st.session_state.last_qa_entry_id = entry_id
            update_query_progress(100, "Answer synthesis failed.")

    if st.session_state.last_answer:
        st.markdown("#### Answer")
        if st.session_state.last_cache_status:
            mode = st.session_state.last_cache_status
            similarity = float(st.session_state.last_cache_similarity)
            if mode == "exact":
                st.success("Served from answer cache (exact match). No model call required.")
            else:
                st.success(
                    f"Served from answer cache ({mode} match, similarity={similarity:.3f}). "
                    "No answer-generation model call required."
                )
        st.write(st.session_state.last_answer)

        question_for_render = st.session_state.last_question or ""
        if _question_requests_visual(question_for_render):
            st.markdown("#### Requested Visualization")
            if st.session_state.last_compute_df.empty:
                st.info(
                    "A chart was requested, but no computed table is available. "
                    "Please refine the question so SQL returns rows."
                )
            else:
                figure, chart_error = _infer_qa_chart(
                    st.session_state.last_compute_df,
                    question_for_render,
                )
                if figure is not None:
                    st.plotly_chart(figure, width="stretch")
                else:
                    st.info(
                        "Could not infer a chart automatically from the returned columns. "
                        f"{chart_error}"
                    )

        if not st.session_state.last_compute_df.empty:
            st.markdown("#### Result Table")
            st.dataframe(st.session_state.last_compute_df.head(500), width="stretch")

        with st.expander("LLM Reasoning Summary (Diagnostics)"):
            st.write(
                st.session_state.last_reasoning_summary
                or "No reasoning summary captured."
            )
            if st.session_state.last_raw_llm_output:
                st.caption("Raw model output")
                st.code(st.session_state.last_raw_llm_output[:8_000], language="json")

        with st.expander("Computation Details"):
            st.caption("Generated SQL")
            st.code(st.session_state.last_sql or "No SQL generated.", language="sql")
            if st.session_state.last_sql_rationale:
                st.caption(f"SQL rationale: {st.session_state.last_sql_rationale}")
            if st.session_state.last_sql_error:
                st.error(f"Compute path error: {st.session_state.last_sql_error}")
            if not st.session_state.last_compute_df.empty:
                st.caption("Result table is shown above.")

        with st.expander("Retrieved Rows Used for Context"):
            retrieval_preview = build_retrieval_preview(df, st.session_state.last_metadatas)
            if retrieval_preview.empty:
                st.info("No retrieval preview available.")
            else:
                st.dataframe(retrieval_preview, width="stretch")

        st.markdown("---")
        _render_feedback_controls(st.session_state.last_qa_entry_id)


def main() -> None:
    _init_session_state()
    st.title("Dataset Explorer + Q&A Agent")
    st.caption(
        "Upload your dataset, inspect key metrics and charts, and ask questions with compute + retrieval support."
    )

    st.sidebar.header("Settings")
    runtime_mode = st.sidebar.radio("Runtime Mode", options=["local", "cloud"], index=0)

    openai_api_key = config.openai_api_key
    openai_model = config.openai_model
    openai_embedding_model = config.openai_embedding_model
    ollama_base_url = config.ollama_base_url
    ollama_model = config.ollama_model
    ollama_embedding_model = config.ollama_embedding_model
    hashing_embedding_dimensions = config.hashing_embedding_dimensions

    st.sidebar.subheader("LLM Settings")
    if runtime_mode == "cloud":
        openai_api_key_input = st.sidebar.text_input(
            "OpenAI API Key",
            value=openai_api_key or "",
            type="password",
        )
        openai_api_key = openai_api_key_input.strip() or None
        openai_model = st.sidebar.text_input("OpenAI Chat Model", value=openai_model)
    else:
        ollama_base_url = st.sidebar.text_input("Ollama Base URL", value=ollama_base_url)
        ollama_model = st.sidebar.text_input("Ollama Model", value=ollama_model)

    st.sidebar.subheader("Embedding Settings")
    embedding_options = ["openai", "ollama", "hashing"]
    default_embedding = "openai" if runtime_mode == "cloud" else "hashing"
    embedding_backend = st.sidebar.selectbox(
        "Embedding Backend",
        options=embedding_options,
        index=embedding_options.index(default_embedding),
        help="Choose embeddings provider independently from the chat model provider.",
    )

    if embedding_backend == "openai":
        if runtime_mode == "local":
            openai_api_key_input = st.sidebar.text_input(
                "OpenAI API Key (Embeddings)",
                value=openai_api_key or "",
                type="password",
            )
            openai_api_key = openai_api_key_input.strip() or None
        openai_embedding_model = st.sidebar.text_input(
            "OpenAI Embedding Model",
            value=openai_embedding_model,
        )
    elif embedding_backend == "ollama":
        if runtime_mode == "cloud":
            ollama_base_url = st.sidebar.text_input("Ollama Base URL", value=ollama_base_url)
        ollama_embedding_model = st.sidebar.text_input(
            "Ollama Embedding Model",
            value=ollama_embedding_model,
        )
    else:
        hashing_embedding_dimensions = st.sidebar.slider(
            "Hashing Embedding Dimensions",
            min_value=128,
            max_value=4096,
            value=int(hashing_embedding_dimensions),
            step=128,
            help="Higher dimensions can improve retrieval quality at the cost of speed and memory.",
        )

    rows_to_index = st.sidebar.slider(
        "Rows to index",
        min_value=1_000,
        max_value=1_000_000,
        value=200_000,
        step=1_000,
    )
    index_batch_size = st.sidebar.slider(
        "Index batch size",
        min_value=1_000,
        max_value=50_000,
        value=10_000,
        step=1_000,
    )
    top_k = st.sidebar.slider("Retrieved rows (top-k)", min_value=2, max_value=30, value=8)
    enable_compute = st.sidebar.checkbox("Enable compute answers (SQL)", value=True)
    enable_rag = st.sidebar.checkbox("Enable RAG retrieval (ChromaDB)", value=True)
    st.sidebar.subheader("Q&A Caching")
    enable_answer_cache = st.sidebar.checkbox(
        "Enable answer cache",
        value=True,
        help="Reuse previous answers for repeated/similar questions to reduce model usage.",
    )
    enable_semantic_cache = st.sidebar.checkbox(
        "Enable semantic cache matching",
        value=True,
        disabled=not enable_answer_cache,
        help="Uses embedding similarity to match paraphrased questions.",
    )
    semantic_cache_threshold = st.sidebar.slider(
        "Semantic cache threshold",
        min_value=0.70,
        max_value=0.99,
        value=0.86,
        step=0.01,
        disabled=not (enable_answer_cache and enable_semantic_cache),
    )
    answer_cache_max_entries = st.sidebar.slider(
        "Max cached answers",
        min_value=20,
        max_value=500,
        value=200,
        step=10,
        disabled=not enable_answer_cache,
    )
    st.sidebar.caption(f"Logs: `{config.log_dir / 'app.log'}`")
    st.sidebar.caption(f"Feedback log: `{config.feedback_log_file}`")

    tab_setup, tab_overview, tab_dashboard, tab_qa, tab_history, tab_logs = st.tabs(
        ["Data Setup", "Overview", "Dashboard", "Q&A Agent", "Q&A History", "Developer Logs"]
    )

    with tab_setup:
        st.subheader("Dataset Setup")
        st.caption(
            "Upload one primary dataset, optional join tables, and an optional reference dictionary."
        )
        uploaded_file = st.file_uploader(
            "Primary dataset (fact/base table)",
            type=["csv", "xlsx", "xls", "json", "parquet"],
            key="primary_dataset_uploader",
        )
        additional_table_files = st.file_uploader(
            "Optional: Additional tables for SQL joins",
            type=["csv", "xlsx", "xls", "json", "parquet"],
            accept_multiple_files=True,
            help="Each file becomes an SQL table. Table names are inferred from file names.",
            key="additional_tables_uploader",
        )
        reference_file = st.file_uploader(
            "Optional: Data dictionary / join-key file",
            type=["txt", "md", "csv", "xlsx", "xls", "json"],
            help="Used as extra context for SQL planning and answer generation.",
            key="reference_context_uploader",
        )

        if st.session_state.df is not None:
            loaded_df = st.session_state.df
            st.success(
                f"Active dataset loaded: {len(loaded_df):,} rows x {loaded_df.shape[1]:,} columns"
            )
            if st.session_state.additional_tables_info:
                st.caption(
                    f"Additional tables loaded: {len(st.session_state.additional_tables_info)}"
                )
            if st.session_state.reference_text:
                st.caption("Reference context loaded.")

    if uploaded_file is None:
        with tab_overview:
            st.info("Upload a dataset in the Data Setup tab to begin.")
        with tab_dashboard:
            st.info("Upload a dataset in the Data Setup tab to view dashboards.")
        with tab_qa:
            st.info("Upload a dataset in the Data Setup tab to ask questions.")
        with tab_history:
            _render_qa_history()
        with tab_logs:
            st.subheader("Application Log Tail")
            if st.button("Refresh Logs"):
                st.rerun()
            log_text = tail_log(config.log_dir / "app.log", max_lines=300)
            st.code(log_text, language="text")
            st.subheader("Feedback Log Tail")
            feedback_log_text = tail_log(config.feedback_log_file, max_lines=200)
            st.code(feedback_log_text, language="json")
        return

    try:
        _load_uploaded_dataset(uploaded_file.name, uploaded_file.getvalue())
    except Exception as exc:
        logger.exception("Dataset load failed: %s", exc)
        st.error(f"Unable to load dataset: {exc}")
        return

    try:
        _load_additional_tables(additional_table_files)
    except Exception as exc:
        logger.exception("Additional tables load failed: %s", exc)
        st.error(f"Unable to load additional tables: {exc}")
        return

    if reference_file is not None:
        try:
            _load_reference_context(reference_file.name, reference_file.getvalue())
        except Exception as exc:
            logger.exception("Reference context load failed: %s", exc)
            st.error(f"Unable to load reference context: {exc}")
    else:
        st.session_state.reference_key = None
        st.session_state.reference_text = ""

    df: pd.DataFrame = st.session_state.df
    additional_tables: dict[str, pd.DataFrame] = st.session_state.additional_tables
    additional_tables_info: list[dict[str, str | int]] = st.session_state.additional_tables_info
    overview = dataset_overview(df)

    with tab_setup:
        metric_cols = st.columns(5)
        metric_cols[0].metric("Rows", f"{overview['row_count']:,}")
        metric_cols[1].metric("Columns", f"{overview['column_count']:,}")
        metric_cols[2].metric("Missing", f"{overview['missing_values']:,}")
        metric_cols[3].metric("Duplicates", f"{overview['duplicate_rows']:,}")
        metric_cols[4].metric("Memory", f"{overview['memory_mb']:.2f} MB")

    with tab_overview:
        st.subheader("Data Preview")
        st.dataframe(df.head(200), width="stretch")

        st.subheader("Column Types")
        dtype_df = df.dtypes.astype(str).reset_index()
        dtype_df.columns = ["column", "dtype"]
        st.dataframe(dtype_df, width="stretch")

        numeric_frame = df.select_dtypes(include="number")
        if not numeric_frame.empty:
            st.subheader("Numeric Summary")
            numeric_profile = dashboard_sample(numeric_frame, max_rows=200_000)
            if len(numeric_profile) < len(numeric_frame):
                st.caption(
                    f"Numeric summary computed on a sampled subset of {len(numeric_profile):,} rows."
                )
            st.dataframe(
                numeric_profile.describe().transpose(),
                width="stretch",
            )

        if st.session_state.reference_text:
            st.subheader("Reference Context Preview")
            st.code(st.session_state.reference_text[:4_000], language="text")

        if additional_tables_info:
            st.subheader("Additional SQL Tables")
            table_info_df = pd.DataFrame(additional_tables_info)
            st.dataframe(table_info_df, width="stretch")

            table_options = [str(item["table_name"]) for item in additional_tables_info]
            selected_table = st.selectbox(
                "Preview additional table",
                options=table_options,
            )
            selected_df = additional_tables.get(selected_table)
            if selected_df is not None:
                st.dataframe(selected_df.head(200), width="stretch")

    with tab_dashboard:
        _render_dashboard(df)

    with tab_qa:
        _render_qa(
            df,
            additional_tables=additional_tables,
            additional_tables_info=additional_tables_info,
            runtime_mode=runtime_mode,
            embedding_backend=embedding_backend,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            openai_embedding_model=openai_embedding_model,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
            ollama_embedding_model=ollama_embedding_model,
            hashing_embedding_dimensions=hashing_embedding_dimensions,
            rows_to_index=rows_to_index,
            index_batch_size=index_batch_size,
            top_k=top_k,
            enable_compute=enable_compute,
            enable_rag=enable_rag,
            enable_answer_cache=enable_answer_cache,
            enable_semantic_cache=enable_semantic_cache,
            semantic_cache_threshold=semantic_cache_threshold,
            answer_cache_max_entries=answer_cache_max_entries,
        )

    with tab_history:
        _render_qa_history()

    with tab_logs:
        st.subheader("Application Log Tail")
        if st.button("Refresh Logs"):
            st.rerun()
        log_text = tail_log(config.log_dir / "app.log", max_lines=300)
        st.code(log_text, language="text")
        st.subheader("Feedback Log Tail")
        feedback_log_text = tail_log(config.feedback_log_file, max_lines=200)
        st.code(feedback_log_text, language="json")


if __name__ == "__main__":
    main()
