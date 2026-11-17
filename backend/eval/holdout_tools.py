"""``python -m eval.holdout_tools`` -- CLI tools to maintain the encrypted holdout (issue #189).

Ports ``gascomply-archive/holdout-tools/{blind_check,review_sheet,drop_rows}.py`` into this
repository, with tests, plus two new commands (``append``, ``export-corpus``) the holdout needs to
grow towards the target in docs/holdout-sizing.md. Every command keeps ``eval.holdout``'s own
privacy rule (``docs/holdout.md``): no command prints a question, a reference answer, a unit id
from a row, or clause text, except ``sheet``, which writes them to its output file only (mode
``0600``), never to the terminal. A load failure prints only the exception's type name, never its
message, which can quote the offending YAML line or row (``eval.gold.parse_gold_set``'s own
contract). The one exception is ``_UsageError`` (a missing database URL, or ``check``'s own bad
arguments): both are fixed sentences with no row data, so ``main`` prints that message in full.

Each command's logic is a pure function, testable on hand-built rows with no database; a small,
separate async function does the one DB read (or none) each command needs. The database is
only ever read here, never written to.

Commands
--------
* ``check (--encrypted F --key K | --yaml F)``: the blind label check (``blind_check.py``).
  Prints only ``hNN [category] FLAGS`` for rows that have at least one flag.
* ``sheet <plaintext.yaml> <out.txt>``: the review sheet (``review_sheet.py``), written to
  ``out`` with mode ``0600``. The terminal sees only the row counts.
* ``drop <plaintext.yaml> hNN...``: remove rows by position (``drop_rows.py``), keeping a
  ``.bak`` and restoring it if the result fails to reload.
* ``append <plaintext.yaml> <new_rows.yaml>``: validate and append new rows, refusing a
  duplicate id or a malformed row without changing the file at all.
* ``export-corpus <dir>``: write ``00-index.txt`` (every document, marked whether it has units)
  plus one file per document that has units, one line per unit, in tree order.
"""

# A CLI whose whole job is a report on stdout (as eval/holdout.py, eval/compare.py).
# ruff: noqa: T201

import argparse
import asyncio
import re
import shutil
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from sqlalchemy import text

from app.core.config import get_settings
from app.core.db import create_engine, create_session_factory, session_scope
from app.repositories.documents import DocumentRepository
from app.schemas.ask import AskOutcome
from app.services.safety import check_emergency
from eval.gold import GoldCategory, GoldRow, load_gold_set, parse_gold_set
from eval.metrics import answer_check


class _UsageError(Exception):
    """A usage mistake whose message is a fixed sentence, safe to print in full.

    Raised only for a missing database URL and for ``check``'s own bad arguments -- both fixed
    strings with no row data, unlike every other error ``main`` catches, which can quote a YAML
    line or a row (issue #189 trap 1) and so prints only ``type(error).__name__``.
    """


# --- check: the blind label check (blind_check.py) -------------------------------------------

_LOW_OVERLAP_THRESHOLD = 0.50
_WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)
_STOP_WORDS = frozenset(
    {
        "và", "của", "là", "các", "có", "được", "theo", "quy", "định", "tại", "này", "cho",
        "với", "không", "phải", "thì", "khi", "đối", "một", "những", "trong", "về", "hoặc",
        "từ", "đến",
    }
)  # fmt: skip
_DOCUMENT_NUMBER_PATTERN = re.compile(r"\b(\d{1,3}[/:]\d{4})\b")
_DIEU_UNIT_PATTERN = re.compile(r"#dieu-(\d+)")
_KHOAN_UNIT_PATTERN = re.compile(r"#khoan-(\d+)")
_DIEU_TEXT_PATTERN = re.compile(r"Điều\s+(\d+)")
_KHOAN_TEXT_PATTERN = re.compile(r"Khoản\s+(\d+)")
_PARENTHETICAL_PATTERN = re.compile(r"\([^)]*\)")

_GET_DOCUMENT_NUMBERS_SQL = text("SELECT id, number FROM documents")
_GET_EXISTING_UNITS_SQL = text("SELECT id FROM units WHERE id = ANY(:ids)")


def _words(value: str) -> list[str]:
    """Return ``value``'s lowercased words, dropping single letters and stop words."""
    return [w for w in _WORD_PATTERN.findall(value.lower()) if len(w) > 1 and w not in _STOP_WORDS]


def _flags_for_unanswerable_row(row: GoldRow) -> list[str]:
    """Return the flags for a ``refusal``/``emergency`` row (blind_check.py's ``else`` branch)."""
    flags: list[str] = []
    if row.source_unit_ids:
        flags.append("UNEXPECTED_SOURCE")
    if row.category is GoldCategory.EMERGENCY and not check_emergency(row.question).escalated:
        flags.append("NOT_ESCALATED")
    return flags


def _reference_mismatch_flags(
    reference: str, sources: Sequence[str], doc_numbers: dict[str, str]
) -> list[str]:
    """Return ``REF_*_MISMATCH`` flags: the answer names a document/Điều/Khoản no source is in."""
    flags: list[str] = []
    numbers_of_sources = {
        doc_numbers.get(unit_id.split("#", 1)[0], "").replace(":", "/") for unit_id in sources
    }
    named_docs = {n.replace(":", "/") for n in _DOCUMENT_NUMBER_PATTERN.findall(reference)}
    if named_docs and not any(
        any(number in candidate for candidate in numbers_of_sources) for number in named_docs
    ):
        flags.append("REF_DOC_MISMATCH")
    dieus = {m for unit_id in sources for m in _DIEU_UNIT_PATTERN.findall(unit_id)}
    named_dieus = set(_DIEU_TEXT_PATTERN.findall(reference))
    if named_dieus and dieus and not (named_dieus & dieus):
        flags.append("REF_DIEU_MISMATCH")
    khoans = {m for unit_id in sources for m in _KHOAN_UNIT_PATTERN.findall(unit_id)}
    named_khoans = set(_KHOAN_TEXT_PATTERN.findall(reference))
    if named_khoans and khoans and not (named_khoans & khoans):
        flags.append("REF_KHOAN_MISMATCH")
    return flags


def _low_overlap_flag(
    reference: str, sources: Sequence[str], contexts: dict[str, str]
) -> list[str]:
    """Return ``[LOW_OVERLAP:x.xx]`` when few of the reference answer's words are in the sources."""
    ref_words = _words(_PARENTHETICAL_PATTERN.sub(" ", reference))
    source_words = set(_words(" ".join(contexts.get(unit_id, "") for unit_id in sources)))
    if not ref_words or not source_words:
        return []
    share = sum(word in source_words for word in ref_words) / len(ref_words)
    return [f"LOW_OVERLAP:{share:.2f}"] if share < _LOW_OVERLAP_THRESHOLD else []


def _flags_for_answerable_row(
    row: GoldRow, existing: frozenset[str], doc_numbers: dict[str, str], contexts: dict[str, str]
) -> list[str]:
    """Return the flags for a ``compliance``/``multi_hop`` row (blind_check.py's ``if`` branch)."""
    flags: list[str] = []
    sources = list(row.source_unit_ids)
    if not sources:
        flags.append("NO_SOURCE")
    missing = [unit_id for unit_id in sources if unit_id not in existing]
    if missing:
        flags.append(f"MISSING_UNIT:{len(missing)}")
    if row.category is GoldCategory.MULTI_HOP and len(sources) < 2:
        flags.append("MULTIHOP_ONE_SOURCE")
    flags.extend(_reference_mismatch_flags(row.reference_answer, sources, doc_numbers))
    if row.answer_must_match or row.answer_must_not_match:
        ok = answer_check(
            AskOutcome.ANSWER,
            row.reference_answer,
            row.answer_must_match,
            row.answer_must_not_match,
        )
        if ok is False:
            flags.append("PATTERN_REF_FAIL")
    flags.extend(_low_overlap_flag(row.reference_answer, sources, contexts))
    return flags


def _flags_for_row(
    row: GoldRow, existing: frozenset[str], doc_numbers: dict[str, str], contexts: dict[str, str]
) -> list[str]:
    """Return every flag code for one row, exactly as ``blind_check.py``'s ``check`` computed them.

    Args:
        row: The gold/holdout row to check, already validated by ``parse_gold_set``.
        existing: ``units.id`` values that exist in the dev database.
        doc_numbers: ``{document_id: number}`` for every document.
        contexts: ``{unit_id: text}`` for every id in ``existing`` (``get_unit_contexts``).

    Returns:
        Flag codes, in the same order ``blind_check.py`` produced them. Empty when the row is
        clean. ``NO_SOURCE``, ``REFUSAL_FLAG``, ``MULTIHOP_ONE_SOURCE`` and ``UNEXPECTED_SOURCE``
        are now unreachable through a row ``GoldRow`` itself validates (its own
        ``_check_category_agrees_with_flags`` already forbids the combination each one checks
        for) -- kept anyway, as defensive, dead code ported faithfully from the archive script.
    """
    flags: list[str] = []
    if (row.category is GoldCategory.REFUSAL) != row.expected_refusal:
        flags.append("REFUSAL_FLAG")
    if row.category in (GoldCategory.COMPLIANCE, GoldCategory.MULTI_HOP):
        flags.extend(_flags_for_answerable_row(row, existing, doc_numbers, contexts))
    else:
        flags.extend(_flags_for_unanswerable_row(row))
    return flags


def check_rows(
    rows: Sequence[GoldRow],
    existing: frozenset[str],
    doc_numbers: dict[str, str],
    contexts: dict[str, str],
) -> list[tuple[str, str, list[str]]]:
    """Return ``(hNN, category, flags)`` for every row that has at least one flag.

    Pure: takes the dev-database data ``_fetch_check_data`` already read, so this is testable
    on hand-built rows with no database at all (issue #189 Scope 1).
    """
    flagged: list[tuple[str, str, list[str]]] = []
    for position, row in enumerate(rows, start=1):
        flags = _flags_for_row(row, existing, doc_numbers, contexts)
        if flags:
            flagged.append((f"h{position:02d}", row.category.value, flags))
    return flagged


async def _fetch_check_data(
    rows: Sequence[GoldRow], database_url: str
) -> tuple[dict[str, str], frozenset[str], dict[str, str]]:
    """Read, read-only, everything ``check_rows`` needs: doc numbers, existing ids, clause text."""
    engine = create_engine(database_url)
    try:
        async with session_scope(create_session_factory(engine)) as session:
            documents = DocumentRepository(session)
            doc_numbers = dict((await session.execute(_GET_DOCUMENT_NUMBERS_SQL)).all())
            all_ids = sorted({unit_id for row in rows for unit_id in row.source_unit_ids})
            existing = (
                frozenset(
                    (await session.execute(_GET_EXISTING_UNITS_SQL, {"ids": all_ids})).scalars()
                )
                if all_ids
                else frozenset()
            )
            contexts = await documents.get_unit_contexts(sorted(existing))
    finally:
        await engine.dispose()
    return doc_numbers, existing, contexts


def _load_check_rows(args: argparse.Namespace) -> tuple[GoldRow, ...]:
    """Load the rows ``check`` runs on, from ``--yaml`` or by decrypting ``--encrypted``."""
    if args.yaml:
        path = Path(args.yaml).expanduser()
        return parse_gold_set(path.read_text(encoding="utf-8"), str(path))
    from eval.holdout import _decrypt  # the runner's own in-memory decryption (issue #178)

    plaintext = _decrypt(Path(args.encrypted).expanduser(), Path(args.key).expanduser())
    return parse_gold_set(plaintext.decode("utf-8"), "holdout")


def _validate_check_args(args: argparse.Namespace) -> None:
    """Raise unless ``check``'s own args are exactly ``--yaml`` xor ``--encrypted``/``--key``."""
    if args.yaml:
        if args.encrypted or args.key:
            raise _UsageError("--yaml cannot be combined with --encrypted/--key")
    elif not (args.encrypted and args.key):
        raise _UsageError("check needs either --yaml, or both --encrypted and --key")


def _check_command(args: argparse.Namespace) -> int:
    """Run ``check``: print only ``rows: N; flagged: M`` then ``hNN [category] FLAGS`` lines."""
    _validate_check_args(args)
    rows = _load_check_rows(args)
    doc_numbers, existing, contexts = asyncio.run(_fetch_check_data(rows, _database_url(args)))
    flagged = check_rows(rows, existing, doc_numbers, contexts)
    print(f"rows: {len(rows)}; flagged: {len(flagged)}")
    for position, category, flags in flagged:
        print(f"{position} [{category}] {' '.join(flags)}")
    return 0


# --- sheet: the review sheet (review_sheet.py) ------------------------------------------------

_SHEET_BODY_LIMIT = 1500
_SHEET_CHILD_LIMIT = 8
_NO_DB_ROW = "KHÔNG TÌM THẤY TRONG DB"

_SHEET_GUIDE = """PHIẾU RÀ NHÃN HOLDOUT -- chỉ bạn đọc. Không mở file này trong VS Code (Claude
chạy ở đó), không dán, không kể lại nội dung cho bất kỳ session Claude nào. Xoá file khi xong.

Với mỗi câu, ghi một dòng vào giấy hoặc ghi chú riêng: hNN OK | hNN BỎ (lý do ngắn).

  emergency  : Có phải tình huống nguy hiểm ĐANG xảy ra (rò gas, cháy, nổ, ngạt...) mà app phải
               hiện cảnh báo gọi 114 / khoá van NGAY không? Không phải -> BỎ.
  refusal    : App có NÊN từ chối không? (Hỏi văn bản không có trong kho 76 văn bản, hỏi ngoài
               phạm vi gas/PCCC, hoặc hỏi tình trạng hiệu lực mà kho không có.) Nếu thật ra kho
               có điều khoản trả lời được -> BỎ.
  compliance : Điều khoản nguồn bên dưới có THỰC SỰ trả lời câu hỏi không? Đáp án mẫu có khớp
               với nội dung điều khoản không? Một trong hai sai -> BỎ.
  multi_hop  : Như compliance, và câu hỏi có thật cần NHIỀU điều khoản không? Nếu điều khoản
               nguồn thiếu hoặc thừa rõ ràng -> BỎ.
  "KHÔNG TÌM THẤY TRONG DB" ở một điều khoản nguồn = nhãn chắc chắn sai -> BỎ.
"""
"""The owner's own audit guide, verbatim from ``review_sheet.py``, minus its last line (a
note specific to one past audit's own known rows, not a general instruction)."""

_GET_CLAUSE_SQL = text(
    "SELECT coalesce(heading, '') AS heading, coalesce(body, '') AS body "
    "FROM units WHERE id = :unit_id"
)
_GET_CLAUSE_CHILDREN_SQL = text(
    "SELECT id, coalesce(heading, '') AS heading, coalesce(body, '') AS body FROM units "
    "WHERE parent_id = :unit_id ORDER BY ordinal NULLS LAST, id"
)


def _render_clause_text(heading: str, body: str, children: Sequence[tuple[str, str, str]]) -> str:
    """Pure: render one clause's own text plus a bounded preview of its children.

    Ports ``review_sheet.py``'s ``clause()`` rendering, reading ``heading``/``body`` as their
    own columns instead of parsing a concatenated, newline-joined ``psql`` line -- same visible
    shape, without that format's fragility.
    """
    own = "\n".join(part for part in (heading, body) if part).strip()
    text_block = own[:_SHEET_BODY_LIMIT] or "(không có nội dung riêng)"
    if not children:
        return text_block
    shown = [
        f"  · {child_id}: {child_heading} {child_body}"[:600]
        for child_id, child_heading, child_body in children[:_SHEET_CHILD_LIMIT]
    ]
    more = len(children) - _SHEET_CHILD_LIMIT
    text_block += "\n  Các khoản/điểm con:\n" + "\n".join(shown)
    if more > 0:
        text_block += f"\n  ... và {more} mục con nữa"
    return text_block


def render_sheet_row(position: int, row: GoldRow, clause_texts: dict[str, str]) -> str:
    """Pure: render one ``hNN`` block of the review sheet (``review_sheet.py``'s loop body)."""
    parts = [
        f"\n{'=' * 78}\nh{position:02d}  [{row.category.value}]  multi_hop={row.multi_hop}  "
        f"expected_refusal={row.expected_refusal}\n\nCÂU HỎI:\n{row.question}\n\n"
        f"ĐÁP ÁN MẪU:\n{row.reference_answer}\n"
    ]
    if row.source_unit_ids:
        parts.append("\nĐIỀU KHOẢN NGUỒN (source_unit_ids):")
        for unit_id in row.source_unit_ids:
            parts.append(f"\n- {unit_id}\n{clause_texts.get(unit_id, _NO_DB_ROW)}")
    else:
        parts.append("\nĐIỀU KHOẢN NGUỒN: (không có)")
    if row.answer_must_match:
        parts.append(f"\n\nanswer_must_match: {[p.pattern for p in row.answer_must_match]}")
    if row.answer_must_not_match:
        parts.append(f"\nanswer_must_not_match: {[p.pattern for p in row.answer_must_not_match]}")
    parts.append("\n\nKẾT LUẬN: hNN OK / BỎ (lý do): ______________________\n")
    return "".join(parts)


def render_sheet(rows: Sequence[GoldRow], clause_texts: dict[str, str]) -> str:
    """Pure: render the whole review sheet, given every cited clause's already-fetched text."""
    parts = [_SHEET_GUIDE]
    parts.extend(
        render_sheet_row(position, row, clause_texts) for position, row in enumerate(rows, start=1)
    )
    return "".join(parts)


async def _fetch_sheet_clauses(rows: Sequence[GoldRow], database_url: str) -> dict[str, str]:
    """Read, read-only, every cited source clause's own text plus its children's."""
    unit_ids = sorted({unit_id for row in rows for unit_id in row.source_unit_ids})
    engine = create_engine(database_url)
    texts: dict[str, str] = {}
    try:
        async with session_scope(create_session_factory(engine)) as session:
            for unit_id in unit_ids:
                own = (await session.execute(_GET_CLAUSE_SQL, {"unit_id": unit_id})).first()
                if own is None:
                    texts[unit_id] = _NO_DB_ROW
                    continue
                children = (
                    await session.execute(_GET_CLAUSE_CHILDREN_SQL, {"unit_id": unit_id})
                ).all()
                texts[unit_id] = _render_clause_text(
                    own.heading, own.body, [(c.id, c.heading, c.body) for c in children]
                )
    finally:
        await engine.dispose()
    return texts


def _sheet_command(args: argparse.Namespace) -> int:
    """Run ``sheet``: write the review sheet to ``out`` (mode 0600), print only the row counts."""
    source, out = Path(args.plaintext).expanduser(), Path(args.out).expanduser()
    rows = load_gold_set(source)
    clause_texts = asyncio.run(_fetch_sheet_clauses(rows, _database_url(args)))
    out.write_text(render_sheet(rows, clause_texts), encoding="utf-8")
    out.chmod(0o600)
    counts = Counter(row.category.value for row in rows)
    print(f"wrote {len(rows)} rows ({dict(counts)}) to {out}")
    return 0


# --- drop: remove rows by position (drop_rows.py) ---------------------------------------------


def drop_rows(rows: Sequence[dict], positions: frozenset[int]) -> list[dict]:
    """Pure: return ``rows`` with the 1-based ``positions`` removed (``drop_rows.py``'s core)."""
    return [row for index, row in enumerate(rows, start=1) if index not in positions]


def _parse_positions(args: Sequence[str]) -> list[int]:
    """Parse ``hNN``-style position arguments into sorted, unique 1-based integers."""
    return sorted({int(arg.lstrip("hH")) for arg in args})


def _drop_command(args: argparse.Namespace) -> int:
    """Run ``drop``: remove rows by position, restoring the original if the result fails to load."""
    path = Path(args.path).expanduser()
    positions = _parse_positions(args.positions)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as error:  # a YAML error quotes the offending line
        print(f"failed: {type(error).__name__}")
        return 2
    if not isinstance(raw, list):
        print("top level is not a list -- nothing changed")
        return 2
    bad = [p for p in positions if not 1 <= p <= len(raw)]
    if bad or not positions:
        print(f"positions must be between h01 and h{len(raw):02d} -- nothing changed")
        return 2
    before = Counter(row.get("category") if isinstance(row, dict) else None for row in raw)
    backup = path.with_name(path.name + ".bak")
    shutil.copy2(path, backup)
    kept = drop_rows(raw, frozenset(positions))
    path.write_text(
        yaml.safe_dump(kept, allow_unicode=True, sort_keys=False, width=100000), encoding="utf-8"
    )
    try:
        loaded = load_gold_set(path)
    except Exception as error:  # never print the message: it can quote the row
        shutil.copy2(backup, path)
        print(f"{type(error).__name__}: result failed to load -- original restored")
        return 2
    after = Counter(row.category.value for row in loaded)
    print(f"dropped {', '.join(f'h{p:02d}' for p in positions)}")
    print(f"before: {len(raw)} rows {dict(before)}")
    print(f"after:  {len(loaded)} rows {dict(after)}  (original kept as {backup.name})")
    return 0


# --- append: validate and add new rows (new, issue #189) --------------------------------------


def append_rows(existing: Sequence[dict], new: Sequence[dict]) -> list[dict]:
    """Pure: concatenate ``existing`` and ``new``, assuming the duplicate-id check already ran."""
    return [*existing, *new]


def _append_command(args: argparse.Namespace) -> int:
    """Run ``append``: validate, refuse a duplicate id or a bad row, else append and re-validate."""
    path, new_path = Path(args.path).expanduser(), Path(args.new_rows).expanduser()
    try:
        existing_raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"failed: {type(error).__name__}")
        return 2
    if not isinstance(existing_raw, list):
        print("top level is not a list -- nothing changed")
        return 2
    try:
        new_raw = yaml.safe_load(new_path.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"failed: {type(error).__name__}")
        return 2
    if not isinstance(new_raw, list):
        print("new rows: top level is not a list -- nothing changed")
        return 2
    try:
        parse_gold_set(new_path.read_text(encoding="utf-8"), "new_rows")
    except Exception as error:  # never print the message: it can quote the row
        print(f"{type(error).__name__}: new rows failed to load -- nothing changed")
        return 2
    existing_ids = {row["id"] for row in existing_raw if isinstance(row, dict) and "id" in row}
    new_ids = {row["id"] for row in new_raw if isinstance(row, dict) and "id" in row}
    duplicates = existing_ids & new_ids
    if duplicates:
        print(f"append refused: {len(duplicates)} id(s) already present -- nothing changed")
        return 2
    before = Counter(row.get("category") if isinstance(row, dict) else None for row in existing_raw)
    backup = path.with_name(path.name + ".bak")
    shutil.copy2(path, backup)
    combined = append_rows(existing_raw, new_raw)
    path.write_text(
        yaml.safe_dump(combined, allow_unicode=True, sort_keys=False, width=100000),
        encoding="utf-8",
    )
    try:
        loaded = load_gold_set(path)
    except Exception as error:  # never print the message: it can quote the row
        shutil.copy2(backup, path)
        print(f"{type(error).__name__}: result failed to load -- original restored")
        return 2
    after = Counter(row.category.value for row in loaded)
    print(f"before: {len(existing_raw)} rows {dict(before)}")
    print(f"after:  {len(loaded)} rows {dict(after)}  (original kept as {backup.name})")
    return 0


# --- export-corpus: write the corpus for the web builder (new, issue #189) --------------------


@dataclass(frozen=True)
class _UnitRow:
    """One ``units`` row as ``export-corpus`` needs it: own coordinates, no recomposed text."""

    id: str
    parent_id: str | None
    ordinal: int
    unit_type: str
    heading: str | None
    body: str
    chapter_no: int | None
    dieu_no: int | None
    khoan_no: int | None
    diem: str | None
    appendix_no: int | None
    item_no: int | None
    appendix_section: str | None


_ROMAN_NUMERAL_VALUES: tuple[tuple[int, str], ...] = (
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
)
_SUBITEM_SUFFIX_PATTERN = re.compile(r"#muc-(\d+)$")
_FORM_CODE_PATTERN = re.compile(r"#mau-([a-z0-9]+)$")


def _roman(value: int) -> str:
    """Return ``value`` in uppercase Roman numerals (an appendix number is always a small one)."""
    remainder, numeral = value, ""
    for amount, symbol in _ROMAN_NUMERAL_VALUES:
        count, remainder = divmod(remainder, amount)
        numeral += symbol * count
    return numeral


def _label_chapter(row: _UnitRow) -> str:
    return f"Chương {row.chapter_no}" if row.chapter_no is not None else "Chương"


def _label_article(row: _UnitRow) -> str:
    return f"Điều {row.dieu_no}" if row.dieu_no is not None else "Điều"


def _label_clause(row: _UnitRow) -> str:
    return f"Khoản {row.khoan_no}" if row.khoan_no is not None else "Khoản"


def _label_point(row: _UnitRow) -> str:
    return f"Điểm {row.diem}" if row.diem else "Điểm"


def _label_appendix(row: _UnitRow) -> str:
    return f"Phụ lục {_roman(row.appendix_no)}" if row.appendix_no else "Phụ lục"


def _label_appendix_section(row: _UnitRow) -> str:
    section = row.appendix_section or ""
    if row.appendix_no:
        return f"Phụ lục {_roman(row.appendix_no)}, Mục {section}"
    return f"Mục {section}"


def _label_appendix_item(row: _UnitRow) -> str:
    return f"Mục {row.item_no}" if row.item_no is not None else "Mục"


def _label_appendix_point(row: _UnitRow) -> str:
    parts = []
    if row.item_no is not None:
        parts.append(f"Mục {row.item_no}")
    if row.diem:
        parts.append(f"Điểm {row.diem}")
    return ", ".join(parts) if parts else "Điểm"


def _label_appendix_subitem(row: _UnitRow) -> str:
    match = _SUBITEM_SUFFIX_PATTERN.search(row.id)
    number = match.group(1) if match else "?"
    prefix = f"Mục {row.item_no}" if row.item_no is not None else "Mục"
    return f"{prefix}.{number}"


def _label_appendix_form(row: _UnitRow) -> str:
    match = _FORM_CODE_PATTERN.search(row.id)
    return f"Mẫu {match.group(1)}" if match else "Mẫu"


_STRUCTURAL_LABELS: dict[str, Callable[[_UnitRow], str]] = {
    "chapter": _label_chapter,
    "article": _label_article,
    "clause": _label_clause,
    "point": _label_point,
    "appendix": _label_appendix,
    "appendix_section": _label_appendix_section,
    "appendix_section_clause": _label_clause,
    "appendix_section_point": _label_point,
    "appendix_item": _label_appendix_item,
    "appendix_point": _label_appendix_point,
    "appendix_subitem": _label_appendix_subitem,
    "appendix_form": _label_appendix_form,
}


def _structural_label(row: _UnitRow) -> str:
    """Return a short Vietnamese structural label for one unit, from its own coordinates."""
    builder = _STRUCTURAL_LABELS.get(row.unit_type)
    return builder(row) if builder is not None else row.unit_type


def _unit_label(row: _UnitRow) -> str:
    """Return one unit's export label: its structural label, plus its heading when it has one."""
    structural = _structural_label(row)
    if row.heading:
        return f"{structural} — {row.heading}" if structural else row.heading
    return structural


def render_index(documents: Sequence[tuple[str, str, str, str]], with_units: frozenset[str]) -> str:
    """Pure: render ``00-index.txt``, one line per document, marked whether it has units."""
    lines = []
    for document_id, number, title_vi, status in documents:
        marker = "có nội dung" if document_id in with_units else "KHÔNG có nội dung trong hệ thống"
        lines.append(f"{document_id}\t{number}\t{title_vi}\t{status}\t{marker}")
    return "\n".join(lines) + "\n"


def _ordered_unit_lines(rows: Sequence[_UnitRow]) -> list[str]:
    """Return one line per unit, parent before child, siblings by ``ordinal`` (issue #189 trap 2).

    Ordering is numeric (``ordinal``), never the id string's own lexical order, so ``khoan-10``
    comes after ``khoan-2``. Every row in ``rows`` gets exactly one line, even a row whose
    ``parent_id`` does not resolve to another row in ``rows`` (treated as a root, same as a
    genuine root whose ``parent_id`` is ``None``) -- issue #189 trap 4's own requirement that
    every unit id appears exactly once.
    """
    by_id = {row.id: row for row in rows}
    children: dict[str | None, list[_UnitRow]] = {}
    for row in rows:
        parent_key = row.parent_id if row.parent_id in by_id else None
        children.setdefault(parent_key, []).append(row)
    for siblings in children.values():
        siblings.sort(key=lambda row: row.ordinal)

    lines: list[str] = []
    visited: set[str] = set()

    def walk(parent_key: str | None) -> None:
        for row in children.get(parent_key, []):
            if row.id in visited:
                continue
            visited.add(row.id)
            lines.append(f"[{row.id}] {_unit_label(row)}: {row.body}")
            walk(row.id)

    walk(None)
    for row in rows:  # defence in depth: a cycle would otherwise silently drop a unit
        if row.id not in visited:
            visited.add(row.id)
            lines.append(f"[{row.id}] {_unit_label(row)}: {row.body}")
    return lines


def render_document(rows: Sequence[_UnitRow]) -> str:
    """Pure: render one document's export file, one line per unit, in tree order."""
    return "\n".join(_ordered_unit_lines(rows)) + "\n"


_GET_EXPORT_DOCUMENTS_SQL = text("SELECT id, number, title_vi, status FROM documents ORDER BY id")
_GET_DOCUMENT_IDS_WITH_UNITS_SQL = text("SELECT DISTINCT document_id FROM units")
_GET_DOCUMENT_UNITS_SQL = text(
    "SELECT id, parent_id, ordinal, unit_type, heading, coalesce(body, '') AS body, "
    "chapter_no, dieu_no, khoan_no, diem, appendix_no, item_no, appendix_section "
    "FROM units WHERE document_id = :document_id"
)


async def _export_corpus(out_dir: Path, database_url: str) -> tuple[int, int]:
    """Write the index and one file per document with units; the database is only ever read.

    Args:
        out_dir: Directory to write into; created if missing.
        database_url: Dev database to read from, read-only.

    Returns:
        ``(documents_with_units, stub_documents)``, for the command's own summary line.
    """
    engine = create_engine(database_url)
    try:
        async with session_scope(create_session_factory(engine)) as session:
            documents = [
                (row.id, row.number, row.title_vi, row.status)
                for row in (await session.execute(_GET_EXPORT_DOCUMENTS_SQL)).all()
            ]
            with_units = frozenset(
                (await session.execute(_GET_DOCUMENT_IDS_WITH_UNITS_SQL)).scalars()
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "00-index.txt").write_text(
                render_index(documents, with_units), encoding="utf-8"
            )
            doc_ids = sorted(
                document_id for document_id, *_rest in documents if document_id in with_units
            )
            for document_id in doc_ids:
                result = await session.execute(
                    _GET_DOCUMENT_UNITS_SQL, {"document_id": document_id}
                )
                rows = [_UnitRow(**dict(row._mapping)) for row in result]
                (out_dir / f"{document_id}.txt").write_text(render_document(rows), encoding="utf-8")
    finally:
        await engine.dispose()
    return len(doc_ids), len(documents) - len(doc_ids)


def _export_corpus_command(args: argparse.Namespace) -> int:
    """Run ``export-corpus``: write the index plus one file per document that has units."""
    with_units, stubs = asyncio.run(
        _export_corpus(Path(args.dir).expanduser(), _database_url(args))
    )
    print(f"wrote {1 + with_units} files ({with_units} documents with units, {stubs} stubs)")
    return 0


# --- CLI ---------------------------------------------------------------------------------------


def _database_url(args: argparse.Namespace) -> str:
    """Return ``--database-url`` if given, else the configured ``DATABASE_URL`` setting."""
    url = getattr(args, "database_url", None) or get_settings().database_url
    if not url:
        raise _UsageError("no database URL: pass --database-url or set DATABASE_URL")
    return url


def _build_parser() -> argparse.ArgumentParser:
    """Build the ``python -m eval.holdout_tools`` argument parser, one subcommand per command."""
    parser = argparse.ArgumentParser(prog="python -m eval.holdout_tools")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="Blind label check (ported from blind_check.py).")
    check.add_argument("--encrypted", help="DER smime-encrypted holdout.")
    check.add_argument("--key", help="Private key; openssl asks for its passphrase.")
    check.add_argument("--yaml", help="Plaintext YAML, for testing on the gold set.")
    check.add_argument("--database-url", default=None, help="Default: the DATABASE_URL setting.")
    check.set_defaults(handler=_check_command)

    sheet = commands.add_parser("sheet", help="Write the label-review sheet (review_sheet.py).")
    sheet.add_argument("plaintext")
    sheet.add_argument("out")
    sheet.add_argument("--database-url", default=None, help="Default: the DATABASE_URL setting.")
    sheet.set_defaults(handler=_sheet_command)

    drop = commands.add_parser("drop", help="Drop rows by their hNN position (drop_rows.py).")
    drop.add_argument("path")
    drop.add_argument("positions", nargs="+")
    drop.set_defaults(handler=_drop_command)

    append = commands.add_parser("append", help="Validate and append new rows.")
    append.add_argument("path")
    append.add_argument("new_rows")
    append.set_defaults(handler=_append_command)

    export = commands.add_parser("export-corpus", help="Export the corpus for the web builder.")
    export.add_argument("dir")
    export.add_argument("--database-url", default=None, help="Default: the DATABASE_URL setting.")
    export.set_defaults(handler=_export_corpus_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one ``eval.holdout_tools`` command.

    Args:
        argv: Command-line arguments, excluding the program name.

    Returns:
        The command's own exit code. Whatever the cause of a failure, only a fixed sentence or
        an exception's type name is printed -- never a question, a reference answer, a unit id
        from a row, or clause text (issue #189 trap 1).
    """
    args = _build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except _UsageError as error:  # a fixed, row-free sentence: safe to print in full
        print(f"failed: {error}")
        return 2
    except Exception as error:  # the message can quote a row or a YAML line
        print(f"failed: {type(error).__name__}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
