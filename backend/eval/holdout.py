"""``python -m eval.holdout`` -- score the encrypted holdout set without leaking it (issue #178).

``data/gold.yaml`` has been read by every session that tuned retrieval and generation, so its
numbers can no longer tell real improvement from fitting to the set (build plan A8). The
holdout is a second, secret set in the same ``GoldRow`` schema, encrypted to the owner's
certificate; this runner is the only thing that ever sees its plaintext, and only the owner runs
it, in their own terminal (``docs/holdout.md``).

Nothing the run emits may name a question, a reference answer, a unit id, an original row id or
a model reply -- not on stdout, not on stderr, not in a log line. How each leak route is shut:

* **Decryption stays in memory.** ``openssl smime -decrypt`` runs as a subprocess whose stdout
  is captured; stdin and stderr are inherited, so openssl itself asks for the passphrase on the
  owner's terminal. The passphrase is never an argument, an environment variable or a Python
  ``input()``, and the plaintext is never written anywhere (``eval.gold.parse_gold_set`` exists
  so no temporary file is needed).
* **Error messages are content.** A YAML error quotes the offending line; a ``GoldRow`` error
  names the row id. Every failure reports the exception's type name and a fixed sentence.
* **Rows are renamed before anything sees them.** ``h01`` ... ``hNN`` replace the ids the
  builder chose, which can name the topic, before the rows reach ``run_evaluation``. Every
  table, RETRIED/FAILED list and log line therefore carries only a position.
* **Logs are an allowlist.** The ``app`` and library loggers do log model replies, unit ids and
  exception messages (``generator``, ``grounding``, ``retriever``), so this runner prints no
  record from them at all. It prints only ``eval.compare``'s own per-row lines, which carry the
  renamed id and numbers, and shows an exception as its type name, never its message or
  traceback.
"""

# A CLI whose whole job is a report on stdout (as ``eval/compare.py``, ``ingest/run.py``).
# ruff: noqa: T201

import argparse
import asyncio
import hashlib
import logging
import subprocess
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from app.core.config import get_settings
from app.schemas.ask import AskOutcome
from eval.compare import (
    EvaluationReport,
    RowScore,
    ask_generation_config,
    format_report,
    run_gold_evaluation,
)
from eval.gold import GoldCategory, GoldRow, parse_gold_set
from eval.metrics import exit_code

EXIT_FAILED = 2
"""Exit code for "the run could not produce a report" (as opposed to ``1``, a report with a
failing threshold)."""

DECRYPT_FAILED = "could not decrypt (wrong key or passphrase, or not run from a terminal)"
SHOW_BANNER = "Do not paste this into any Claude session: a question that reaches one is burned."
_SHOW_NEEDS_TERMINAL = (
    "--show prints the holdout's questions and needs a terminal on stdin and stdout"
)
_ROW_ID_FORMAT = "h{:02d}"
_LOG_ALLOWLIST = frozenset({"eval.compare"})
"""The only loggers whose records are printed: their messages are built from the (renamed) row
id and numbers alone."""


class _Failure(Exception):  # noqa: N818 - a control-flow signal for ``main``, not an error type
    """A run-ending condition whose ``str`` is a fixed sentence, safe to print as it is."""


# --- loading ---------------------------------------------------------------------------------


def load_holdout(encrypted: Path, key: Path) -> tuple[GoldRow, ...]:
    """Decrypt, parse and rename the holdout, all in memory.

    Args:
        encrypted: The DER ``smime`` envelope.
        key: The private key; if passphrase-protected, openssl prompts on the terminal.

    Returns:
        The rows in file order, with ``id`` replaced by their position (``h01`` ...).

    Raises:
        _Failure: With a fixed message, if the file cannot be decrypted or parsed.
    """
    plaintext = _decrypt(encrypted, key)
    try:
        rows = parse_gold_set(plaintext.decode("utf-8"), "holdout")
    except Exception as error:  # the message would quote the row or line
        raise _Failure(f"{type(error).__name__}: holdout file failed to load") from None
    if not rows:
        raise _Failure("holdout file failed to load: it has no rows")
    return tuple(
        row.model_copy(update={"id": _ROW_ID_FORMAT.format(position)})
        for position, row in enumerate(rows, start=1)
    )


def _decrypt(encrypted: Path, key: Path) -> bytes:
    """Return the decrypted bytes, captured from openssl's stdout (never written to a file)."""
    command = [
        "openssl",
        "smime",
        "-decrypt",
        "-binary",
        "-inform",
        "DER",
        "-in",
        str(encrypted),
        "-inkey",
        str(key),
    ]
    try:
        # stdin and stderr are inherited on purpose: openssl prompts for the passphrase on the
        # controlling terminal, and that terminal is the boundary (it cannot be scripted).
        completed = subprocess.run(command, stdout=subprocess.PIPE, check=False)
    except OSError:
        raise _Failure(DECRYPT_FAILED) from None
    if completed.returncode != 0:
        raise _Failure(DECRYPT_FAILED)
    return completed.stdout


def _digest(encrypted: Path) -> str:
    """First 12 hex digits of the encrypted file's SHA-256, to tie two runs to one file version."""
    try:
        return hashlib.sha256(encrypted.read_bytes()).hexdigest()[:12]
    except OSError:
        raise _Failure("could not read the encrypted file") from None


# --- logging ---------------------------------------------------------------------------------


class _ContentFreeFormatter(logging.Formatter):
    """Format a record as level, logger and message; an exception becomes its type name only."""

    def format(self, record: logging.LogRecord) -> str:
        line = f"{record.levelname:<8} {record.name} {record.getMessage()}"
        if record.exc_info and record.exc_info[0] is not None:
            line += f" [{record.exc_info[0].__name__}]"
        return line


@contextmanager
def _content_free_logging() -> Iterator[None]:
    """Replace the root handlers with one that prints only allowlisted, content-free records."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_ContentFreeFormatter())
    handler.addFilter(lambda record: record.name in _LOG_ALLOWLIST)

    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    root.handlers, root.level = [handler], logging.INFO
    logging.captureWarnings(True)  # a warning's text goes to the dropped ``py.warnings`` logger
    try:
        yield
    finally:
        logging.captureWarnings(False)
        root.handlers, root.level = saved_handlers, saved_level


# --- --show ----------------------------------------------------------------------------------


def _failed_checks(report: EvaluationReport) -> dict[str, list[str]]:
    """Map each row id that failed any check to the names of the checks it failed."""
    failed: dict[str, list[str]] = {}

    def add(row_id: str, check: str) -> None:
        failed.setdefault(row_id, []).append(check)

    for score in report.answerable:
        for check in _answerable_failures(score):
            add(score.row_id, check)
    for row_id in report.over_answered_row_ids:
        add(row_id, "refusal (answered instead of refusing)")
    for row_id in report.unescalated_row_ids:
        add(row_id, "safety detection (not escalated)")
    for row_id in report.failed_row_ids:
        add(row_id, "not evaluated (raised twice, excluded from every metric)")
    return failed


def _answerable_failures(score: RowScore) -> list[str]:
    checks: list[str] = []
    if score.hybrid_strict_recall < 1.0:
        checks.append("recall@k (strict)")
    if score.hybrid_outcome is not AskOutcome.ANSWER:
        checks.append("effective answer (refused)")
    elif score.hybrid_citation_precision is not None and score.hybrid_citation_precision < 1.0:
        checks.append("citation accuracy")
    if score.hybrid_answer_check is False:
        checks.append("answer check")
    return checks


def _show(report: EvaluationReport, rows: Sequence[GoldRow]) -> None:
    """Print every failing row's question and reference answer, for the owner's eyes only."""
    failed = _failed_checks(report)
    print()
    print(SHOW_BANNER)
    if not failed:
        print("No row failed any check.")
        return
    for row in rows:
        if row.id not in failed:
            continue
        print()
        print(f"{row.id} [{row.category.value}] failed: {'; '.join(failed[row.id])}")
        print(f"  question: {row.question}")
        print(f"  reference answer: {row.reference_answer}")


# --- CLI -------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Score the encrypted holdout and print the report.

    Args:
        argv: Command-line arguments, excluding the program name.

    Returns:
        ``eval.compare``'s exit code (``0`` / ``1``) for a finished run, ``2`` if the run could
        not produce a report. Whatever the cause, only a fixed sentence or an exception type
        name is printed.
    """
    args = _parse_args(argv)
    if args.show and not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(_SHOW_NEEDS_TERMINAL, file=sys.stderr)
        return EXIT_FAILED
    try:
        with _content_free_logging():
            return asyncio.run(_amain(args))
    except _Failure as failure:
        print(failure, file=sys.stderr)
    except Exception as error:  # its message can carry a question or a reply
        print(f"the run failed: {type(error).__name__}", file=sys.stderr)
    return EXIT_FAILED


async def _amain(args: argparse.Namespace) -> int:
    encrypted, key = Path(args.encrypted), Path(args.key)
    digest = _digest(encrypted)
    rows = load_holdout(encrypted, key)
    counts = " ".join(
        f"{category.value}={sum(row.category is category for row in rows)}"
        for category in GoldCategory
    )
    print(f"holdout sha256: {digest}")
    print(f"rows: {counts}", flush=True)

    settings = get_settings()
    report = await run_gold_evaluation(
        rows,
        settings,
        args.database_url or settings.database_url,
        ask_generation_config(settings),
    )
    if report is None:
        return EXIT_FAILED

    print(format_report(report))
    if args.show:
        _show(report, rows)
    return exit_code(report.verdicts)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m eval.holdout",
        description="Score the encrypted holdout set in memory; print only row positions.",
    )
    parser.add_argument("--encrypted", required=True, help="The DER smime-encrypted holdout.")
    parser.add_argument(
        "--key", required=True, help="The private key; openssl asks for its passphrase."
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres URL to read from (default: the DATABASE_URL setting/env var).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Owner only: after the report, print each failing row's question and reference "
        "answer. Refused unless stdin and stdout are terminals.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
