"""Tests for the offline ``LLMProvider`` double (issue #90)."""

import asyncio
import json
from datetime import date

from app.llm.base import LLMProvider
from app.llm.mock import MockLLM
from app.schemas.retrieval import Passage, Why
from app.services.generator import _build_user_prompt, _Source


def _generate(user: str, system: str = "ignored") -> dict:
    reply = asyncio.run(MockLLM().generate(system=system, user=user))
    return json.loads(reply)


def test_mock_llm_implements_the_llm_provider_interface():
    assert isinstance(MockLLM(), LLMProvider)


def test_no_unit_id_marker_at_all_refuses():
    reply = _generate("NGỮ CẢNH:\n\nCÂU HỎI: không có gì để trích dẫn ở đây")

    assert reply == {"refusal": True, "answer": None, "citations": []}


def test_one_marker_is_cited():
    reply = _generate("[unit_id: doc-a#dieu-1]\nVăn bản.\n\nCÂU HỎI: hỏi gì đó")

    assert reply["refusal"] is False
    assert reply["answer"]
    assert reply["citations"] == [{"unit_id": "doc-a#dieu-1"}]


def test_markers_are_cited_in_the_order_they_appear():
    user = "[unit_id: doc-a#dieu-1]\nA.\n\n[unit_id: doc-b#dieu-2]\nB.\n\nCÂU HỎI: x"

    reply = _generate(user)

    assert [c["unit_id"] for c in reply["citations"]] == ["doc-a#dieu-1", "doc-b#dieu-2"]


def test_at_most_three_markers_are_cited():
    user = "\n\n".join(f"[unit_id: doc-{i}#dieu-1]\ntext {i}." for i in range(6))

    reply = _generate(user)

    assert len(reply["citations"]) == 3


def test_a_repeated_marker_is_cited_once():
    user = "[unit_id: doc-a#dieu-1]\nA.\n\n[unit_id: doc-a#dieu-1]\nA again.\n\nCÂU HỎI: x"

    reply = _generate(user)

    assert [c["unit_id"] for c in reply["citations"]] == ["doc-a#dieu-1"]


def test_deterministic_for_the_same_input():
    user = "[unit_id: doc-a#dieu-1]\nA.\n\nCÂU HỎI: x"

    assert _generate(user) == _generate(user)


def test_the_json_schema_argument_is_accepted_and_ignored():
    # MockLLM's reply shape never depends on json_schema; passing one must not raise.
    reply = asyncio.run(
        MockLLM().generate(
            system="s", user="[unit_id: doc-a#dieu-1]\nA.", json_schema={"type": "object"}
        )
    )
    assert json.loads(reply)["citations"] == [{"unit_id": "doc-a#dieu-1"}]


def test_text_that_merely_mentions_unit_id_without_the_bracket_marker_is_not_read():
    # Only the exact "[unit_id: ...]" line generator.py writes is a real marker; a citation
    # or a question that happens to contain the words "unit_id" must not be mistaken for one.
    reply = _generate("Người dùng hỏi về khái niệm unit_id trong hệ thống là gì.")

    assert reply == {"refusal": True, "answer": None, "citations": []}


# --- bound to generator.py's own prompt shape, not a hand-written string (issue #101) --------
#
# Every test above hand-writes its own "[unit_id: X]\ntext" prompt -- convenient, but it is
# exactly what let the marker regex ship broken for two issues (#94, then this one): a
# hand-written string with the marker alone on its own line cannot catch a regex that only
# matches a marker alone on its own line, because _build_user_prompt never writes one that way.
# It always appends the source's label on the *same* line as the marker
# (f"[unit_id: {passage.unit_id}] {label}"), which is the one shape these tests below build,
# through _build_user_prompt itself, so a future change to that line's shape breaks this test
# instead of shipping another silent refusal.


def _passage(unit_id: str) -> Passage:
    return Passage(
        unit_id=unit_id,
        document_id=unit_id.split("#", 1)[0],
        dieu_no=53,
        text="Cửa hàng bán lẻ LPG chai phải có diện tích tối thiểu là 12 m2.",
        similarity=0.8,
        hop=0,
        score=0.8,
        why=(Why.SIMILAR,),
    )


def _source(unit_id: str) -> _Source:
    document_id = unit_id.split("#", 1)[0]
    return _Source(
        unit_id=unit_id,
        document_id=document_id,
        document_number="87/2018/NĐ-CP",
        doc_type="decree",
        dieu_no=53,
        khoan_no=None,
        effective=date(2018, 8, 1),
        possibly_superseded=False,
    )


def test_extracts_the_marker_from_a_real_generator_built_prompt():
    unit_id_a = "gas-decree-87-2018#dieu-53"
    unit_id_b = "gas-decree-87-2018#dieu-54#khoan-1"
    passages = (_passage(unit_id_a), _passage(unit_id_b))
    sources = {unit_id_a: _source(unit_id_a), unit_id_b: _source(unit_id_b)}

    prompt = _build_user_prompt("diện tích tối thiểu cửa hàng gas", passages, (), sources)
    reply = _generate(prompt)

    assert reply["refusal"] is False
    assert [c["unit_id"] for c in reply["citations"]] == [unit_id_a, unit_id_b]


def test_a_source_with_no_dieu_or_khoan_still_matches_through_its_label():
    # _label appends only what a source actually has (see generator._label); with neither
    # dieu_no nor khoan_no, the label is at its shortest -- still on the same line as the
    # marker, still must not defeat extraction.
    unit_id = "gas-decree-87-2018"
    passages = (_passage(unit_id),)
    source = _Source(
        unit_id=unit_id,
        document_id=unit_id,
        document_number="87/2018/NĐ-CP",
        doc_type="decree",
        dieu_no=None,
        khoan_no=None,
        effective=None,
        possibly_superseded=False,
    )

    prompt = _build_user_prompt("x", passages, (), {unit_id: source})
    reply = _generate(prompt)

    assert [c["unit_id"] for c in reply["citations"]] == [unit_id]


def test_extracts_every_marker_from_a_structured_article_passage():
    """Issue #187: each focus unit's line now ends in ``: <body>``, not just a label.

    ``_UNIT_ID_MARKER`` anchors on the ``[unit_id: ...]`` prefix/suffix alone, so it
    must still find every marker when what follows on the line is a unit's own body
    text rather than a bare label -- this is the shape ``_build_user_prompt`` now
    writes for an article passage's focus units.
    """
    article = "gas-decree-87-2018#dieu-8"
    khoan_1 = f"{article}#khoan-1"
    diem_a = f"{khoan_1}#diem-a"
    passage = Passage(
        unit_id=article,
        document_id="gas-decree-87-2018",
        dieu_no=8,
        text="Điều 8. Điều kiện đối với thương nhân kinh doanh mua bán khí",
        similarity=0.8,
        hop=0,
        score=0.8,
        why=(Why.SIMILAR,),
        focus_units=(khoan_1, diem_a),
    )
    sources = {
        article: _source(article),
        khoan_1: _Source(
            unit_id=khoan_1,
            document_id="gas-decree-87-2018",
            document_number="87/2018/NĐ-CP",
            doc_type="decree",
            dieu_no=8,
            khoan_no=1,
            effective=None,
            possibly_superseded=False,
            body="Điều kiện gồm:",
            parent_id=article,
            ordinal=1,
        ),
        diem_a: _Source(
            unit_id=diem_a,
            document_id="gas-decree-87-2018",
            document_number="87/2018/NĐ-CP",
            doc_type="decree",
            dieu_no=8,
            khoan_no=1,
            effective=None,
            possibly_superseded=False,
            body="Là thương nhân được thành lập theo quy định của pháp luật.",
            parent_id=khoan_1,
            ordinal=1,
        ),
    }

    prompt = _build_user_prompt("x", (passage,), (), sources)
    reply = _generate(prompt)

    assert [c["unit_id"] for c in reply["citations"]] == [article, khoan_1, diem_a]
