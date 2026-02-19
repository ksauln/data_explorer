from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.logging_utils import append_jsonl_record, tail_log


class LoggingUtilsTests(unittest.TestCase):
    def test_append_jsonl_record_writes_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "feedback.jsonl"
            append_jsonl_record(log_path, {"entry_id": "abc", "feedback": "up"})
            append_jsonl_record(log_path, {"entry_id": "def", "feedback": "down"})

            with log_path.open("r", encoding="utf-8") as file:
                lines = file.readlines()

            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            second = json.loads(lines[1])
            self.assertEqual(first["entry_id"], "abc")
            self.assertEqual(second["feedback"], "down")

    def test_tail_log_handles_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "does_not_exist.log"
            output = tail_log(log_path)
            self.assertIn("does not exist", output)


if __name__ == "__main__":
    unittest.main()
