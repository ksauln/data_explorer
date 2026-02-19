from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    log_dir: Path
    chroma_dir: Path
    feedback_log_file: Path
    default_collection: str
    openai_api_key: str | None
    openai_model: str
    openai_embedding_model: str
    ollama_base_url: str
    ollama_model: str
    ollama_embedding_model: str
    hashing_embedding_dimensions: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        log_dir = Path(os.getenv("LOG_DIR", "logs"))
        feedback_log_file_env = os.getenv("FEEDBACK_LOG_FILE", "").strip()
        return cls(
            log_dir=log_dir,
            chroma_dir=Path(os.getenv("CHROMA_DIR", "chroma_db")),
            feedback_log_file=(
                Path(feedback_log_file_env)
                if feedback_log_file_env
                else log_dir / "feedback.jsonl"
            ),
            default_collection=os.getenv("CHROMA_COLLECTION", "dataset_rows"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL",
                "text-embedding-3-small",
            ),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            ollama_embedding_model=os.getenv(
                "OLLAMA_EMBEDDING_MODEL",
                "mxbai-embed-large",
            ),
            hashing_embedding_dimensions=int(
                os.getenv("HASHING_EMBEDDING_DIMENSIONS", "512")
            ),
        )
