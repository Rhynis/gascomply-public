"""Tests for the encrypted-holdout runner (issue #178).

``eval.holdout`` decrypts a secret gold set in memory and must print nothing that names a
question, a reference answer, a unit id, an original row id or a model reply. Every fake set
below puts the marker ``zqx`` inside each of those, so one assertion -- ``"zqx" not in output``
-- covers all of them.

Each test makes its own throwaway key pair in ``tmp_path``; nothing here reads a real key or the
real holdout. Leaks are caught at the file-descriptor level (``capfd``), not with ``caplog``:
the owner's terminal sees file descriptors, and a handler-level filter is invisible to
``caplog``.

The tests need ``openssl`` (LibreSSL 3 on macOS, OpenSSL 3 on CI); a missing binary fails them
rather than skipping them, so CI can never pass without having run them. The ones that score
rows also need Postgres (``TEST_DATABASE_URL``), like ``tests/test_compare.py``.
"""

import hashlib
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from app.core.config import get_settings
from app.llm.base import LLMProvider, LLMUnavailableError, LLMUnreachableError
from eval import compare, holdout
from eval.gold import GoldCategory, load_gold_set, parse_gold_set

BACKEND_DIR = Path(__file__).resolve().parent.parent
LEAK = "zqx"
"""Every question, reference answer, unit id, row id and model reply below contains this."""

_FAKE_SET = """\
- id: compliance-zqx-area-rule
  category: compliance
  question: "zqx-question-1: diện tích tối thiểu của cửa hàng là bao nhiêu?"
  reference_answer: "zqx-answer-1: 12 m2."
  source_unit_ids: ["zqx-doc#dieu-7"]
- id: multi-hop-zqx-two-clauses
  category: multi_hop
  question: "zqx-question-2: hai điều khoản nào cùng áp dụng?"
  reference_answer: "zqx-answer-2: điều 7 và điều 8."
  source_unit_ids: ["zqx-doc#dieu-7", "zqx-doc#dieu-8#khoan-1"]
  multi_hop: true
- id: refusal-zqx-out-of-corpus
  category: refusal
  question: "zqx-question-3: giá gas hôm nay là bao nhiêu?"
  reference_answer: "zqx-answer-3: ngoài phạm vi."
  expected_refusal: true
- id: emergency-zqx-smell
  category: emergency
  question: "zqx-question-4: ngửi thấy mùi gas nồng nặc"
  reference_answer: "zqx-answer-4: gọi 114."
"""
_ORIGINAL_IDS = ("compliance-zqx-area-rule", "multi-hop-zqx-two-clauses")


def _load_sibling_module(filename: str) -> ModuleType:
    """Load a sibling test module by path (as ``tests/test_compare.py`` does)."""
    path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(f"holdout_sibling_{path.stem}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive only
        raise ImportError(f"cannot load sibling test module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _openssl(*args: str, cwd: Path) -> None:
    subprocess.run(["openssl", *args], cwd=cwd, check=True, capture_output=True)


def _make_holdout(
    directory: Path, plaintext: str = _FAKE_SET, *, name: str = "a", passphrase: str | None = None
) -> tuple[Path, Path]:
    """Create a throwaway key pair in ``directory`` and encrypt ``plaintext`` to it.

    Returns:
        ``(encrypted, key)`` paths. The key is unprotected (``-nodes``) unless ``passphrase``
        is given.
    """
    directory.mkdir(parents=True, exist_ok=True)
    key, cert = directory / f"{name}-key.pem", directory / f"{name}-cert.pem"
    protection = ["-passout", f"pass:{passphrase}"] if passphrase else ["-nodes"]
    _openssl(
        "req", "-x509", "-newkey", "rsa:2048", *protection, "-days", "1", "-subj", "/CN=test",
        "-keyout", key.name, "-out", cert.name,
        cwd=directory,
    )  # fmt: skip
    plain, encrypted = directory / f"{name}-plain.yaml", directory / f"{name}.p7m"
    plain.write_text(plaintext, encoding="utf-8")
    _openssl(
        "smime", "-encrypt", "-aes256", "-binary", "-outform", "DER",
        "-in", plain.name, "-out", encrypted.name, cert.name,
        cwd=directory,
    )  # fmt: skip
    plain.unlink()  # the plaintext must not sit next to the ciphertext, even in a test
    return encrypted, key


@pytest.fixture(autouse=True)
def _require_openssl() -> None:
    assert shutil.which("openssl"), "openssl is required: these tests must run in CI, not skip"


@pytest.fixture(autouse=True)
def _no_ambient_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep ``get_settings`` deterministic and offline (as ``tests/test_compare.py`` does)."""
    for name in ("DATABASE_URL", "EMBEDDER", "LLM_PROVIDER", "RERANKER", "RERANK_TOP_N"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def bare_root_logger() -> Iterator[None]:
    """Take pytest's capture handlers off the root logger.

    A record nothing else handles then reaches ``logging.lastResort`` -> stderr, as it would in
    the owner's terminal. Without this, pytest's own handlers would swallow a leaking record and
    the leak tests would prove nothing.
    """
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers = []
    yield
    root.handlers = saved


# --- decrypting and renaming (no database) --------------------------------------------------


def test_the_premise_a_yaml_error_quotes_the_line_and_a_row_error_names_the_row():
    with pytest.raises(Exception, match=LEAK):
        parse_gold_set("- id: [zqx-unclosed\n", "holdout")
    with pytest.raises(Exception, match="compliance-zqx-bad"):
        parse_gold_set(
            "- id: compliance-zqx-bad\n  category: compliance\n  question: q\n"
            "  reference_answer: a\n",
            "holdout",
        )


def test_parse_gold_set_matches_load_gold_set_on_the_same_text(tmp_path: Path):
    path = tmp_path / "gold.yaml"
    path.write_text(_FAKE_SET, encoding="utf-8")

    assert parse_gold_set(_FAKE_SET, str(path)) == load_gold_set(path)


def test_load_holdout_returns_the_rows_renamed_by_position_and_writes_no_file(tmp_path: Path):
    encrypted, key = _make_holdout(tmp_path)
    before = sorted(tmp_path.rglob("*"))

    rows = holdout.load_holdout(encrypted, key)

    assert [row.id for row in rows] == ["h01", "h02", "h03", "h04"]
    assert [row.category for row in rows] == [
        GoldCategory.COMPLIANCE,
        GoldCategory.MULTI_HOP,
        GoldCategory.REFUSAL,
        GoldCategory.EMERGENCY,
    ]
    assert rows[0].question.startswith("zqx-question-1")  # the content itself is intact
    assert sorted(tmp_path.rglob("*")) == before, "decrypting left a file behind"


def test_the_wrong_key_exits_2_and_prints_no_marker(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    encrypted, _ = _make_holdout(tmp_path / "owner")
    _, wrong_key = _make_holdout(tmp_path / "stranger", name="b")

    code = holdout.main(["--encrypted", str(encrypted), "--key", str(wrong_key)])

    captured = capfd.readouterr()
    assert code == 2
    assert LEAK not in captured.out + captured.err


def test_malformed_yaml_exits_2_reports_the_type_only_and_prints_no_marker(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    encrypted, key = _make_holdout(tmp_path, "- id: [zqx-unclosed\n  question: zqx-line\n")

    code = holdout.main(["--encrypted", str(encrypted), "--key", str(key)])

    captured = capfd.readouterr()
    assert code == 2
    assert "holdout file failed to load" in captured.err
    assert "Error" in captured.err  # the exception's type name
    assert LEAK not in captured.out + captured.err


def test_a_row_that_fails_validation_exits_2_and_does_not_name_the_row(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    encrypted, key = _make_holdout(
        tmp_path, "- id: compliance-zqx-bad\n  category: nonsense\n  question: zqx\n"
    )

    code = holdout.main(["--encrypted", str(encrypted), "--key", str(key)])

    captured = capfd.readouterr()
    assert code == 2
    assert "ValidationError: holdout file failed to load" in captured.err
    assert LEAK not in captured.out + captured.err


def test_show_without_a_terminal_exits_2_before_decrypting_anything(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    encrypted, key = _make_holdout(tmp_path)

    code = holdout.main(["--encrypted", str(encrypted), "--key", str(key), "--show"])

    captured = capfd.readouterr()
    assert code == 2
    assert "needs a terminal" in captured.err
    assert "sha256" not in captured.out
    assert LEAK not in captured.out + captured.err


def test_a_passphrase_protected_key_with_no_terminal_exits_2_at_once(tmp_path: Path):
    """No controlling terminal, so openssl cannot prompt: it must fail, never hang or block.

    ``start_new_session=True`` drops the controlling terminal and ``stdin`` is ``/dev/null``,
    so this can never block on a passphrase prompt in a developer's own terminal.
    """
    encrypted, key = _make_holdout(tmp_path, passphrase="correct horse")

    completed = subprocess.run(
        [sys.executable, "-m", "eval.holdout", "--encrypted", str(encrypted), "--key", str(key)],
        cwd=BACKEND_DIR,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        start_new_session=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 2
    assert holdout.DECRYPT_FAILED in completed.stderr
    assert LEAK not in completed.stdout + completed.stderr


# --- scoring through the shared path (database, stub LLM) ----------------------------------


class _StubLLM(LLMProvider):
    """Returns ``reply``, or raises ``error``, for every call (cf. ``tests/test_compare.py``)."""

    def __init__(self, reply: str = "", error: Exception | None = None) -> None:
        self.reply, self.error = reply, error

    async def generate(
        self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
    ) -> str:
        if self.error is not None:
            raise self.error
        if json_schema is not None and "supported" in json_schema.get("properties", {}):
            return json.dumps({"supported": True})
        return self.reply


def _reply(**fields: Any) -> str:
    return json.dumps({"refusal": False, "answer": "câu trả lời", "citations": [], **fields})


_LLM_BEHAVIOURS = {
    # the warning "dropped a fabricated citation -- 'zqx-doc#dieu-7'" names a unit id
    "cites-a-unit-id": lambda: _StubLLM(
        _reply(answer="zqx-model-answer", citations=[{"unit_id": "zqx-doc#dieu-7"}])
    ),
    # the warning "model reply was not valid JSON: 'zqx-model-reply...'" quotes the reply
    "replies-with-junk": lambda: _StubLLM("zqx-model-reply, not JSON"),
    # "the LLM provider was unavailable: zqx..." carries the exception's message
    "provider-unavailable": lambda: _StubLLM(error=LLMUnavailableError("zqx-question zqx-reply")),
    # logger.exception in run_evaluation prints the traceback, message included
    "raises-with-content": lambda: _StubLLM(error=RuntimeError("zqx-question zqx-model-reply")),
}


@pytest.fixture
def ingested_db(migrated_db: str, tmp_path: Path) -> str:
    """The migrated test database holding ``tests/test_gold.py``'s fixture corpus."""
    from ingest.run import main as ingest_main

    corpus = _load_sibling_module("test_gold.py")._build_fixture_corpus(tmp_path / "fixture")
    code = ingest_main(
        [
            "--corpus", str(corpus),
            "--database-url", migrated_db,
            "--references", str(tmp_path / "references.csv"),
        ]
    )  # fmt: skip
    assert code == 0, "fixture corpus failed to ingest"
    return migrated_db


@pytest.mark.usefixtures("bare_root_logger")
@pytest.mark.parametrize("behaviour", _LLM_BEHAVIOURS)
def test_a_full_run_prints_only_row_positions_whatever_the_model_does(
    behaviour: str,
    ingested_db: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
):
    encrypted, key = _make_holdout(tmp_path / "holdout")
    monkeypatch.setattr(compare, "_create_llm", lambda settings: _LLM_BEHAVIOURS[behaviour]())

    code = holdout.main(
        ["--encrypted", str(encrypted), "--key", str(key), "--database-url", ingested_db]
    )

    captured = capfd.readouterr()
    output = captured.out + captured.err
    assert code in (0, 1)
    assert "h01" in output and "h02" in output
    assert LEAK not in output
    for original_id in _ORIGINAL_IDS:
        assert original_id not in output


class _UnreachableOnceLLM(LLMProvider):
    """Raises ``LLMUnreachableError`` on its first call, then answers normally (issue #185).

    Mirrors ``tests/test_compare.py``'s own double of the same shape -- this one lives here
    too because it is what drives a real wait through ``eval.holdout``'s own log allowlist.
    """

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    async def generate(
        self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
    ) -> str:
        self.calls += 1
        if self.calls == 1:
            raise LLMUnreachableError("zqx-question zqx-model-reply")
        if json_schema is not None and "supported" in json_schema.get("properties", {}):
            return json.dumps({"supported": True})
        return self.reply


@pytest.mark.usefixtures("bare_root_logger")
def test_a_wait_during_a_run_prints_only_the_row_position(
    ingested_db: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
):
    """Issue #185, Scope 4 and 6: the wait/resume log lines pass the holdout's own allowlist.

    ``LLM_PROVIDER=ollama`` is what makes ``run_gold_evaluation`` build a waiter at all
    (Scope 2); ``_build_ollama_waiter`` itself is monkeypatched to a fake, instant one, so
    this never touches a real network or sleeps for real (only the real-socket test in
    ``tests/test_compare.py`` does that). The row is scored through the retry, and both of
    ``run_evaluation``'s wait log lines (carrying only the renamed row id and the exception's
    type name -- issue #185, trap 2) must still pass the holdout's own content-free,
    allowlisted logging the same as every other line in the report.
    """
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    get_settings.cache_clear()
    encrypted, key = _make_holdout(tmp_path / "holdout")
    stub = _UnreachableOnceLLM(_reply())
    monkeypatch.setattr(compare, "_create_llm", lambda settings: stub)

    async def _instant_waiter() -> bool:
        return True

    monkeypatch.setattr(compare, "_build_ollama_waiter", lambda base_url, **kwargs: _instant_waiter)

    code = holdout.main(
        ["--encrypted", str(encrypted), "--key", str(key), "--database-url", ingested_db]
    )

    captured = capfd.readouterr()
    output = captured.out + captured.err
    assert code in (0, 1)
    assert stub.calls > 1  # the first call raised; later calls (including a retry) succeeded
    assert "provider waits: 1," in output
    # The wait/resume log lines themselves (``eval.compare``'s own logger, already
    # allowlisted) must have reached stdout/stderr, and must name only the renamed row id
    # and the exception's type name -- never a question, a reply or the original row id.
    assert "waiting for Ollama after row h01 (LLMUnreachableError), up to 300 s" in output
    assert "Ollama answered again after" in output
    assert "h01" in output
    assert LEAK not in output
    for original_id in _ORIGINAL_IDS:
        assert original_id not in output


def test_the_report_starts_with_the_file_digest_and_the_rows_per_category(
    ingested_db: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
):
    encrypted, key = _make_holdout(tmp_path / "holdout")
    monkeypatch.setattr(compare, "_create_llm", lambda settings: _StubLLM(_reply()))

    holdout.main(["--encrypted", str(encrypted), "--key", str(key), "--database-url", ingested_db])

    lines = capfd.readouterr().out.splitlines()
    digest = hashlib.sha256(encrypted.read_bytes()).hexdigest()[:12]
    assert lines[0] == f"holdout sha256: {digest}"
    assert lines[1] == "rows: compliance=1 multi_hop=1 refusal=1 emergency=1"
    assert lines[2].startswith("GraphRAG vs naive vector search")  # format_report follows


def test_a_run_writes_no_file(ingested_db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    encrypted, key = _make_holdout(tmp_path / "holdout")
    monkeypatch.setattr(compare, "_create_llm", lambda settings: _StubLLM(_reply()))
    monkeypatch.chdir(tmp_path)
    before = sorted(tmp_path.rglob("*"))

    holdout.main(["--encrypted", str(encrypted), "--key", str(key), "--database-url", ingested_db])

    assert sorted(tmp_path.rglob("*")) == before
