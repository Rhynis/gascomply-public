"""Tests for the deterministic document-number guard and citation verifier."""

import asyncio
import json
from datetime import date

from app.llm.base import LLMProvider
from app.schemas.ask import Citation
from app.services.grounding import (
    GroundingTrace,
    SupportCheck,
    document_numbers_are_grounded,
    filter_supported_citations,
    in_force_claims_without_in_force_document,
    legal_numbers,
)


def _check(source_texts: dict[str, str] | None = None) -> SupportCheck:
    return SupportCheck(question="hỏi", answer="đáp", source_texts=source_texts or {})


def _citation(unit_id: str, number: str = "105/2025/NĐ-CP") -> Citation:
    return Citation(
        unit_id=unit_id,
        document_id=unit_id.split("#", 1)[0],
        document_number=number,
        effective=date(2025, 1, 1),
    )


class _ScriptedVerifier(LLMProvider):
    def __init__(self, verdicts: list[bool]) -> None:
        self.verdicts = verdicts
        self.calls: list[dict[str, object]] = []

    async def generate(self, *, system: str, user: str, json_schema: dict | None = None) -> str:
        self.calls.append({"system": system, "user": user, "json_schema": json_schema})
        return json.dumps({"supported": self.verdicts[len(self.calls) - 1]})


def test_document_number_guard_rejects_full_and_bare_unmentioned_numbers() -> None:
    citation = _citation("doc-a#dieu-1")
    sources = {citation.unit_id: "Điều khoản chỉ nói về an toàn."}
    assert not document_numbers_are_grounded(
        "Nghị định 157/2022/NĐ-CP thay thế văn bản cũ.", (citation,), sources
    )
    # The bare form still only counts as a document number behind a legal-instrument noun
    # (the PR review's fix for item 1); "Nghị định" here is that noun, not "Văn bản" (which
    # is not one -- a generic "document" reference never grounds or ungrounds a number).
    assert not document_numbers_are_grounded(
        "Nghị định 157/2022 đã thay thế.", (citation,), sources
    )


def test_document_number_guard_accepts_cited_document_and_text_numbers() -> None:
    own = _citation("doc-a#dieu-1", "157/2022/NĐ-CP")
    cited = _citation("doc-b#dieu-1", "105/2025/NĐ-CP")
    assert document_numbers_are_grounded("Nghị định 157/2022 có hiệu lực.", (own,), {})
    assert document_numbers_are_grounded(
        "QCVN 08:2019/BCT áp dụng.",
        (cited,),
        {cited.unit_id: "Theo QCVN 08:2019/BCT, cơ sở phải..."},
    )
    assert legal_numbers("Nghị định 157/2022/NĐ-CP và QCVN 08:2019/BCT") == {
        "157/2022/NĐ-CP",
        "QCVN08:2019/BCT",
    }


def test_document_number_guard_does_not_match_a_different_document_suffix() -> None:
    citation = _citation("doc-a#dieu-1", "105/2025/NĐ-CP")
    assert not document_numbers_are_grounded("Thông tư 105/2025/TT-BCT áp dụng.", (citation,), {})


def test_document_number_guard_rejects_an_ungrounded_standard_number() -> None:
    citation = _citation("doc-a#dieu-1")
    sources = {citation.unit_id: "Điều khoản chỉ nói về an toàn."}
    assert not document_numbers_are_grounded("QCVN 08:2019/BCT áp dụng.", (citation,), sources)


def test_a_bare_date_is_never_read_as_a_document_number() -> None:
    r"""A date written "NN/M/YYYY" must never become a document number (the PR review).

    ``_BARE_NUMBER`` used to match the ``\d{1,4}/\d{4}`` substring of any date (e.g. the
    "7/2025" inside "01/7/2025"), so a correct answer naming the correct effective-date
    clause was refused because no clause contains "7/2025" as an external document number.
    """
    assert legal_numbers("Luật có hiệu lực từ ngày 01/7/2025.") == frozenset()
    assert legal_numbers("Áp dụng từ tháng 7/2025.") == frozenset()
    assert legal_numbers("ngày 15/5/2025") == frozenset()


def test_a_bare_number_after_a_legal_noun_is_still_a_document_number() -> None:
    assert legal_numbers("Nghị định 87/2018 quy định...") == {"87/2018"}
    assert legal_numbers("Thông tư số 36/2025 hướng dẫn...") == {"36/2025"}


def test_a_full_form_number_is_still_caught_with_no_noun_in_front() -> None:
    """The noun requirement only applies to the bare ``NNN/YYYY`` form.

    A full form (its own ``/<suffix>``, or the ``QCVN``/``TCVN`` standard form) is still
    caught with nothing in front of it at all, exactly as before this fix (the PR review's
    own trap).
    """
    assert legal_numbers("157/2022/NĐ-CP đã hết hiệu lực.") == {"157/2022/NĐ-CP"}
    assert legal_numbers("QCVN 08:2019/BCT áp dụng.") == {"QCVN08:2019/BCT"}


def test_the_real_effective_date_answer_is_grounded_once_the_bare_number_is_not_a_date() -> None:
    """End to end, with the real clause text the PR review reproduced the bug against.

    ``pccc-law-2024#dieu-54#khoan-1``: the wrongly-refused form now grounds correctly, and
    the always-grounded form (no slash at all) still does.
    """
    citation = _citation("pccc-law-2024#dieu-54#khoan-1", "55/2024/QH15")
    sources = {
        citation.unit_id: (
            "Luật này có hiệu lực thi hành từ ngày 01 tháng 7 năm 2025, trừ trường hợp quy "
            "định tại khoản 4 Điều này."
        )
    }
    assert document_numbers_are_grounded(
        "Luật PCCC có hiệu lực từ ngày 01/7/2025.", (citation,), sources
    )
    assert document_numbers_are_grounded(
        "Luật PCCC có hiệu lực từ ngày 01 tháng 7 năm 2025.", (citation,), sources
    )


def test_status_guard_refuses_unknown_and_expired_document_claims() -> None:
    statuses = {
        "87/2018/NĐ-CP": "in_force",
        "41/2011/TT-BCT": "unknown",
        "QCVN 08:2012/BCT": "expired",
    }
    assert in_force_claims_without_in_force_document(
        "Thông tư 41/2011/TT-BCT vẫn còn áp dụng.", statuses
    ) == {"41/2011/TT-BCT"}
    assert in_force_claims_without_in_force_document(
        "QCVN 08:2012/BCT đang có hiệ u lực.", statuses
    ) == {"QCVN08:2012/BCT"}
    assert not in_force_claims_without_in_force_document(
        "Nghị định 87/2018/NĐ-CP về kinh doanh khí vẫn còn hiệu lực, nhưng đã bị sửa đổi "
        "bởi Nghị định 17/2020/NĐ-CP.",
        statuses,
    )


def test_status_guard_leaves_negated_claims_and_unrelated_stub_numbers_alone() -> None:
    statuses = {"41/2011/TT-BCT": "unknown", "27/2001/QH10": "expired"}
    assert not in_force_claims_without_in_force_document(
        "Luật số 27/2001/QH10 đã hết hiệu lực. Thông tư 41/2011/TT-BCT chỉ được nhắc tới.",
        statuses,
    )


def test_status_guard_detects_the_bare_con_hieu_luc_phrasing() -> None:
    """Regression test for a real bug found reviewing this branch.

    ``_IN_FORCE_CLAIM`` used to require a literal "có" between "còn"/"vẫn"/"đang" and
    "hiệu lực", so ordinary Vietnamese for "still in force" with no "có" at all --
    "còn hiệu lực", "vẫn còn hiệu lực" -- silently matched nothing, even though "còn hiệu
    lực" is one of the issue's own explicitly required phrasings (distinct from "còn áp
    dụng", which did match before this fix).
    """
    statuses = {"41/2011/TT-BCT": "unknown"}
    assert in_force_claims_without_in_force_document(
        "Thông tư 41/2011/TT-BCT còn hiệu lực.", statuses
    ) == {"41/2011/TT-BCT"}
    assert in_force_claims_without_in_force_document(
        "Thông tư 41/2011/TT-BCT vẫn còn hiệu lực.", statuses
    ) == {"41/2011/TT-BCT"}


def test_status_guard_scopes_an_in_force_claim_to_its_own_clause() -> None:
    """A trailing "nhưng ..." clause naming a different document must not inherit the claim.

    Same sentence as the issue's own "kept" acceptance case, but with the first document's
    status set to ``unknown`` instead of ``in_force`` -- proving the clause split that makes
    the acceptance case pass is scoping the claim correctly (to the document actually named
    as "vẫn còn hiệu lực"), not merely failing to detect anything at all. If the amending
    document's number leaked into this claim, the result would be ``{"87/2018/NĐ-CP"}`` only
    (never also naming "17/2020/NĐ-CP", which the sentence never claims is in force at all).
    """
    statuses = {"87/2018/NĐ-CP": "unknown"}
    assert in_force_claims_without_in_force_document(
        "Nghị định 87/2018/NĐ-CP về kinh doanh khí vẫn còn hiệu lực, nhưng đã bị sửa đổi "
        "bởi Nghị định 17/2020/NĐ-CP.",
        statuses,
    ) == {"87/2018/NĐ-CP"}


def test_verifier_keeps_supported_drops_unsupported_and_preserves_order() -> None:
    citations = (_citation("doc-a#dieu-1"), _citation("doc-a#dieu-2"))
    llm = _ScriptedVerifier([True, False])
    trace = GroundingTrace()
    result = asyncio.run(filter_supported_citations(_check(), citations, llm, trace))
    assert [citation.unit_id for citation in result] == ["doc-a#dieu-1"]
    assert trace.decisions == [("doc-a#dieu-1", True), ("doc-a#dieu-2", False)]
    assert all(call["json_schema"] for call in llm.calls)
    assert len(trace.verifier_latencies_ms) == 2


class _MalformedReplyVerifier(LLMProvider):
    """A scripted, adversarial provider that never returns the verifier's own schema.

    Stands in for a real provider misbehaving (truncated output, a hosted model ignoring the
    JSON-schema ``format``, Ollama returning something unexpected under load) -- the one path
    ``MockLLM``'s always-clean ``{"supported": true}`` reply never exercises.
    """

    def __init__(self, raw: str) -> None:
        self.raw = raw

    async def generate(self, *, system: str, user: str, json_schema: dict | None = None) -> str:
        return self.raw


def test_a_reply_that_is_not_json_fails_closed_and_drops_the_citation() -> None:
    citation = _citation("doc-a#dieu-1")
    trace = GroundingTrace()
    llm = _MalformedReplyVerifier("khong phai JSON dau")

    result = asyncio.run(filter_supported_citations(_check(), (citation,), llm, trace))

    assert result == ()
    assert trace.decisions == [(citation.unit_id, False)]


def test_a_reply_missing_the_supported_field_fails_closed_and_drops_the_citation() -> None:
    citation = _citation("doc-a#dieu-1")
    trace = GroundingTrace()
    # Valid JSON, but the generator's own schema rather than the verifier's -- exactly what a
    # legacy/offline double that ignores json_schema might send back.
    llm = _MalformedReplyVerifier(
        json.dumps({"refusal": False, "answer": "ok", "citations": [{"unit_id": citation.unit_id}]})
    )

    result = asyncio.run(filter_supported_citations(_check(), (citation,), llm, trace))

    assert result == ()
    assert trace.decisions == [(citation.unit_id, False)]
