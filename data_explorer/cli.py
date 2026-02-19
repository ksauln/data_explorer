from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import unittest
from pathlib import Path

TEST_MODULES = (
    "tests.test_answer_format_utils",
    "tests.test_cache_utils",
    "tests.test_compute_engine",
    "tests.test_data_utils",
    "tests.test_qa_engine",
    "tests.test_logging_utils",
    "tests.test_vector_store",
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="data-explorer",
        description="Run Data Explorer QA. Tests run first by default.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip unit tests before launching Streamlit.",
    )
    parser.add_argument(
        "--test-verbosity",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="Verbosity for unit test output.",
    )
    parser.add_argument(
        "streamlit_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to `streamlit run`.",
    )
    return parser.parse_args(argv)


def _assert_python_312() -> None:
    current = sys.version_info
    if (current.major, current.minor) != (3, 12):
        raise SystemExit(
            f"Python 3.12 is required. Detected {current.major}.{current.minor}. "
            "Use a 3.12 environment and reinstall dependencies."
        )


def _run_unit_tests(verbosity: int) -> bool:
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    for module_name in TEST_MODULES:
        suite.addTests(loader.loadTestsFromName(module_name))

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result.wasSuccessful()


def _resolve_app_path() -> Path:
    app_module = importlib.import_module("app")
    return Path(app_module.__file__).resolve()


def _launch_streamlit(app_path: Path, streamlit_args: list[str]) -> int:
    extra_args = list(streamlit_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    command = [sys.executable, "-m", "streamlit", "run", str(app_path), *extra_args]
    completed = subprocess.run(command, check=False)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _assert_python_312()

    if not args.skip_tests:
        tests_ok = _run_unit_tests(verbosity=args.test_verbosity)
        if not tests_ok:
            return 1

    app_path = _resolve_app_path()
    return _launch_streamlit(app_path, args.streamlit_args)
