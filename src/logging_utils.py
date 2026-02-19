from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


def setup_logging(log_dir: Path, log_level: int = logging.INFO) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("data_explorer")

    if logger.handlers:
        return logger

    logger.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = RotatingFileHandler(
        filename=log_dir / "app.log",
        maxBytes=2_000_000,
        backupCount=5,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def tail_log(log_path: Path, max_lines: int = 200) -> str:
    if not log_path.exists():
        return "Log file does not exist yet."

    with log_path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    return "".join(lines[-max_lines:]) if lines else "Log file is empty."


def append_jsonl_record(log_path: Path, record: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")
