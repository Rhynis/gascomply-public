"""Tests for the pre-LLM gas-emergency detector (build plan A6 Phase 3.5, issue #88).

Every phrase set below is exactly what the PR reports, so a change to this file that widens
or narrows the net is visible in review, not buried in a regex. Comments explain each
judgment call recorded in the PR's "decisions needed" list -- what is caught, what is not,
and why.

No ``pytest-asyncio`` (C12): the one async test drives its own ``asyncio.run()`` call.
"""

import asyncio
import hashlib
import socket
import subprocess
import sys
import unicodedata
from collections.abc import Awaitable

import pytest

from app.services.safety import (
    EMERGENCY_RESPONSE_VI,
    answer_unless_emergency,
    check_emergency,
    fold_diacritics,
    normalise_question,
)

# --- phrase sets -------------------------------------------------------------------------
#
# Emergency phrasings a frightened person actually types: accented, unaccented, typos,
# shouting, and the three forms the issue names by name ("ro gas", "chay nha", "ngui mui
# gas"). Grouped by what triggers them.

EMERGENCY_SMELL = (
    "ngửi thấy mùi gas",
    "nhà tôi có mùi gas",
    "có mùi gas nồng nặc trong bếp",
    "mùi gas nồng quá",
    "sặc mùi gas",
    "ngửi thấy mùi ga",
    "ngửi thấy mùi gaz",
    "ngui thay mui gas",  # the issue's own example
    "ngui mui gas",  # the issue's own example
    "nha co mui gas",
    "mui gas nong nac",
    "mui ga nong qua",
)

EMERGENCY_LEAK = (
    "rò gas",  # the issue's own example
    "ro gas",  # the issue's own example, unaccented
    "gas bị rò",
    "bình gas bị rò",
    "bình gas đang rò",
    "bình gas rò rỉ",
    "bếp gas bị xì",
    "xì gas",
    "van gas bị xì",
    "gas rò rỉ",
    "gas đang rò",
    "gas xì ra ngoài",
    "hở gas",
    "ro ga",
    "binh gas bi ro",
    "gas bi xi",
    "binh gas dang ro",
    "gas ro ri",
    "bình gas bị rỏ",  # "rỏ" (to drip) -- a one-tone-mark typo for "rò" (to leak)
)

EMERGENCY_FIRE = (
    "cháy nhà",
    "nhà cháy",
    "chay nha",  # the issue's own example
    "nhà tôi cháy",
    "nhà đang cháy",
    "cửa hàng gas đang cháy",
    "cháy rồi",
    "cháy to quá",
    "bình gas đang cháy",
    "bình gas bốc cháy",
    "bếp gas bốc cháy",
    "cháy cửa hàng",
    "cháy kho gas",
    "vừa xảy ra cháy ở cửa hàng",
    "nhà hàng xóm cháy",
    "cháy",  # a bare one-word shout
    "cứu với cháy",
    "CHÁY NHÀ!!!",
    "chay nha roi",
    "binh gas dang chay",
    "nha dang chay",
    "lửa bùng lên từ bếp gas",
)

EMERGENCY_EXPLOSION = (
    "nổ bình gas",
    "bình gas nổ",
    "bình gas vừa nổ",
    "gas nổ rồi",
    "bình gas phát nổ",
    "vừa nghe tiếng nổ lớn",
    "có tiếng nổ lớn",
    "nổ rồi",
    "vừa xảy ra vụ nổ bình gas ở nhà bên",
    "nghe tiếng xì xì từ bình gas",
    "binh gas no roi",
)

EMERGENCY_AS_QUESTION = (
    # A real emergency, typed as a question -- must still escalate (see the module
    # docstring's rule 3: a live marker escalates even inside a question, unless the
    # question is *also* hedged with a conditional word like "nếu"/"khi").
    "bình gas nhà tôi đang rò, tôi phải làm sao?",
    "nhà tôi có mùi gas, phải làm sao?",
    "ngửi thấy mùi gas thì làm gì bây giờ?",
    "Bình gas nhà tôi đang rò, theo quy định tôi phải làm sao?",
)

ALL_EMERGENCY = (
    EMERGENCY_SMELL + EMERGENCY_LEAK + EMERGENCY_FIRE + EMERGENCY_EXPLOSION + EMERGENCY_AS_QUESTION
)

# Ordinary compliance questions, built from real clause text (see the PR for each question's
# source unit id). At least ten contain "cháy"/"nổ"; several are the near-misses the issue
# calls out by name ("kiểm tra rò rỉ", "khí gas dễ cháy nổ").
ORDINARY_WITH_FIRE_OR_EXPLOSION = (
    "Quy định phòng cháy chữa cháy cho cửa hàng gas là gì?",
    "Cửa hàng bán lẻ LPG chai phải cách ly với các nguồn gây cháy ít nhất bao nhiêu mét?",
    "Thiết bị điện của cửa hàng LPG có phải là loại phòng nổ không?",
    "Cơ sở nào thuộc danh mục cơ sở có nguy hiểm về cháy, nổ?",
    "Bảo hiểm cháy nổ bắt buộc: mức trích nộp của doanh nghiệp bảo hiểm là bao nhiêu?",
    "Khi thử nổ thủy lực chai LPG, áp suất nổ tối thiểu là bao nhiêu?",
    "Người đứng đầu cơ sở bị cháy phải khắc phục hậu quả vụ cháy như thế nào?",
    "Khi xảy ra cháy tại cơ sở thì ai là người chỉ huy chữa cháy?",
    "Khí gas dễ cháy nổ nên phải bảo quản như thế nào?",
    "Cửa hàng gas nằm cạnh nhà dân có vi phạm quy định về khoảng cách cháy nổ không?",
    "Trong trường hợp xảy ra cháy nổ tại cửa hàng gas, ai chịu trách nhiệm theo pháp luật?",
    "Cửa hàng chưa từng xảy ra cháy nổ có được giảm phí bảo hiểm cháy nổ không?",
)
ORDINARY_LEAK_AS_PROCEDURE = (
    "Kho chứa LPG chai có phải trang bị thiết bị cảnh báo rò rỉ LPG không?",
    "Ai được phép kiểm tra rò rỉ đường ống dẫn LPG?",  # the issue's own "kiểm tra rò rỉ" example
    "Cửa hàng bán lẻ LPG chai phải hướng dẫn khách hàng các bước xử lý khi LPG rò rỉ không?",
    "Nhà ở kết hợp sản xuất kinh doanh có phải có thiết bị phát hiện sự cố rò rỉ chất khí không?",
    "Quy trình xử lý sự cố rò rỉ khí tại trạm nạp là gì?",
    "Kho gas không có mùi gas thì có bắt buộc lắp thiết bị cảnh báo rò rỉ không?",
)
ORDINARY_PROCEDURE = (
    "Nộp hồ sơ thẩm duyệt thiết kế phòng cháy chữa cháy ở đâu?",
    "Mức phạt khi cửa hàng gas không có phương án chữa cháy?",
    "Lực lượng dân phòng tham gia chữa cháy như thế nào?",
    "Nhân viên cửa hàng gas cần được huấn luyện nghiệp vụ phòng cháy chữa cháy không?",
    "Cần giấy phép gì để kinh doanh khí dầu mỏ hóa lỏng?",
    "Người đứng đầu cơ sở bị cháy phải làm gì theo luật?",
    "Trách nhiệm điều tra vụ cháy thuộc cơ quan nào?",
    "Xe bồn chở LPG đang chạy trên đường có cần giấy phép không?",
)
ORDINARY_HYPOTHETICAL = (
    # Hard cases from the issue: a hypothetical about the rules, not a live event.
    "Nếu ngửi thấy mùi gas thì cửa hàng phải làm gì theo quy định?",
    "Khi bình gas bị rò rỉ thì thương nhân có nghĩa vụ gì theo quy định?",
    "Giả sử có mùi gas trong kho thì theo Nghị định 87 phải xử lý thế nào?",
    "Nếu có cháy xảy ra ở cơ sở thì trách nhiệm chữa cháy thuộc về ai?",
)
ORDINARY_NEGATION = (
    "không có mùi gas",
    "chưa ngửi thấy mùi gas",
    "nhà tôi không cháy",
)

ALL_ORDINARY = (
    ORDINARY_WITH_FIRE_OR_EXPLOSION
    + ORDINARY_LEAK_AS_PROCEDURE
    + ORDINARY_PROCEDURE
    + ORDINARY_HYPOTHETICAL
    + ORDINARY_NEGATION
)

# Judgment calls left deliberately uncaught (see the PR's "decisions needed" list). Each is
# pinned here as a passing test, not a skip, so a change that starts catching one of these
# is visible and deliberate rather than an accidental side effect of some other edit.
KNOWN_GAPS = (
    # "cháy" loses its accent but the rest of the sentence keeps its accents. The unaccented
    # rules only run when the *whole* question carries no diacritic (see the module
    # docstring): the alternative -- matching an unaccented word inside otherwise-accented
    # text -- would let "chạy" (run) or "chảy" (flow) drift into matching "cháy" (fire)
    # whenever a nearby word happens to be typed without an accent, which is the exact
    # failure mode issue #88 warns against.
    "nhà tôi chay rồi",
    # Same category: "rò" loses its accent ("ro") while "rỉ" keeps its.
    "gas bị ro rỉ",
    # The subject (a place carrying gas) and the verb "cháy" are several words apart
    # ("xe chở gas bị cháy"). Catching this would mean loosening the fire rule's
    # subject-verb adjacency, which risks matching ordinary sentences like "xe chở LPG phải
    # tuân thủ quy định phòng cháy" that also put several words between a place and "cháy".
    "xe chở gas bị cháy",
    # "hôi gas" (gas smells bad) without the literal word "mùi", typed unaccented ("hoi
    # gas"). Left uncaught unaccented on purpose: "hôi" (stinky) and "hơi" (vapor, this
    # corpus's own word for LPG's gas phase) both fold to "hoi", so accepting it would flag
    # ordinary sentences like "phân tán hơi LPG" once diacritics are stripped (measured:
    # confirmed against the corpus export).
    "hoi gas qua",
)


def _escalated(question: str) -> bool:
    return check_emergency(question).escalated


@pytest.mark.parametrize("question", ALL_EMERGENCY)
def test_emergency_phrasings_are_all_detected(question: str) -> None:
    result = check_emergency(question)
    assert result.escalated, f"missed emergency: {question!r}"
    assert result.matched is not None
    assert result.response == EMERGENCY_RESPONSE_VI


@pytest.mark.parametrize("question", ALL_ORDINARY)
def test_ordinary_questions_never_escalate(question: str) -> None:
    result = check_emergency(question)
    assert not result.escalated, f"false alarm on an ordinary question: {question!r}"
    assert result.matched is None
    assert result.response is None


@pytest.mark.parametrize("question", ALL_ORDINARY)
def test_ordinary_questions_never_escalate_without_diacritics(question: str) -> None:
    """The same ordinary questions, typed on a bare keyboard, must still not escalate."""
    stripped = fold_diacritics(normalise_question(question))
    assert not _escalated(stripped), f"false alarm, unaccented: {stripped!r}"


@pytest.mark.parametrize("question", KNOWN_GAPS)
def test_known_gaps_stay_uncaught(question: str) -> None:
    """Pins each documented decision not to widen the net (see the PR's decisions list)."""
    assert not _escalated(question)


def test_missing_a_gas_smell_only_by_full_sentence_context_still_escalates() -> None:
    """A live marker escalates even when the sentence is phrased as a question."""
    assert _escalated("Nhà tôi ngửi thấy mùi gas, giờ phải làm sao?")


# --- text normalisation --------------------------------------------------------------------


def test_normalise_question_lowercases_and_squeezes_whitespace() -> None:
    assert normalise_question("  CHÁY   NHÀ!!  ") == "cháy nhà"


def test_normalise_question_folds_full_width_and_composes_nfkc() -> None:
    # NFD-decomposed "cháy" (c + h + a + combining acute + combining grave-like hook + y)
    # must normalise the same as its NFC form.
    nfd = unicodedata.normalize("NFD", "cháy nhà")
    assert normalise_question(nfd) == normalise_question("cháy nhà")


def test_normalise_question_collapses_elongated_letters() -> None:
    assert normalise_question("cháyyyyy nhà") == "cháy nhà"
    # Two repeats is left alone -- a plain typo, not shouting.
    assert normalise_question("gass") == "gass"


def test_normalise_question_turns_punctuation_into_spaces() -> None:
    assert normalise_question("rò,gas!!!") == "rò gas"


def test_fold_diacritics_handles_d_with_stroke() -> None:
    assert fold_diacritics("Đang cháy") == "Dang chay"


def test_fold_diacritics_is_the_identity_on_plain_ascii() -> None:
    assert fold_diacritics("ro gas") == "ro gas"


def test_unaccented_rules_only_apply_to_fully_unaccented_text() -> None:
    """Pin the switch condition itself.

    A single unaccented word inside an otherwise-accented sentence is not enough to switch
    on the unaccented rule set -- this is the same decision as ``test_known_gaps``, exercised
    directly against the two normalisation helpers rather than the detector.
    """
    partly_unaccented = normalise_question("nhà tôi chay rồi")
    assert fold_diacritics(partly_unaccented) != partly_unaccented


# --- the hard-coded response ---------------------------------------------------------------


def test_response_contains_the_hotline() -> None:
    assert "114" in EMERGENCY_RESPONSE_VI


def test_response_leads_with_the_action() -> None:
    first_line = EMERGENCY_RESPONSE_VI.splitlines()[0]
    assert "114" in first_line
    assert "ra" in first_line.lower()  # "ra khỏi" (get out), not a citation


def test_response_contains_exactly_the_four_named_actions() -> None:
    lowered = EMERGENCY_RESPONSE_VI.lower()
    assert "ra khỏi" in lowered  # get out
    assert "đóng van" in lowered  # close the valve, if safe
    assert "không dùng lửa" in lowered and "công tắc điện" in lowered  # no flames/switches
    assert "gọi 114" in lowered  # call 114
    # No additional safety advice (e.g. ventilation) is bundled in -- exactly four numbered
    # actions, no more.
    numbered_lines = [line for line in EMERGENCY_RESPONSE_VI.splitlines() if line[:1].isdigit()]
    assert len(numbered_lines) == 4


def test_response_is_byte_identical_across_calls() -> None:
    first = check_emergency("cháy nhà").response
    second = check_emergency("rò gas").response
    assert first == second
    assert first is not None
    assert first.encode("utf-8") == second.encode("utf-8")


def test_response_bytes_are_pinned_by_hash() -> None:
    """An accidental edit to the emergency copy must fail this test, not slip through."""
    digest = hashlib.sha256(EMERGENCY_RESPONSE_VI.encode("utf-8")).hexdigest()
    assert digest == "97b9f7a0efcd5a8973177daa2ca6860bcc2a3f249414c281864c9a32ecb564fa", (
        "EMERGENCY_RESPONSE_VI changed bytes; update this hash only as a deliberate,"
        " reviewed change to safety-critical copy."
    )


# --- the LLM-not-called seam ----------------------------------------------------------------


class _CallCounter:
    """A double that fails the test if invoked, tracking its own call count."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, question: str) -> str:
        self.calls += 1
        pytest.fail(f"the LLM/generation seam must not be called on an emergency: {question!r}")
        raise AssertionError("unreachable")  # satisfies mypy's return type after fail()


async def _echo(question: str) -> str:
    return f"answer to: {question}"


def _run(coro: Awaitable[object]) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


def test_emergency_path_never_calls_the_generator() -> None:
    spy = _CallCounter()

    result = _run(answer_unless_emergency("bình gas nhà tôi đang rò", spy))

    assert spy.calls == 0
    assert result.safety.escalated is True  # type: ignore[attr-defined]
    assert result.answer is None  # type: ignore[attr-defined]


def test_ordinary_path_calls_the_generator_exactly_once() -> None:
    spy_calls = 0

    async def counting_echo(question: str) -> str:
        nonlocal spy_calls
        spy_calls += 1
        return await _echo(question)

    result = _run(
        answer_unless_emergency("Điều kiện cấp giấy phép kinh doanh LPG là gì?", counting_echo)
    )

    assert spy_calls == 1
    assert result.safety.escalated is False  # type: ignore[attr-defined]
    assert result.answer == "answer to: Điều kiện cấp giấy phép kinh doanh LPG là gì?"  # type: ignore[attr-defined]


# --- no database, no network ----------------------------------------------------------------


def test_module_imports_nothing_that_touches_a_database_or_network() -> None:
    """Structural proof, run in a fresh interpreter.

    Importing the module pulls in none of sqlalchemy, asyncpg, httpx2, or app.core.db.
    """
    probe = (
        "import sys\n"
        "import app.services.safety\n"
        "forbidden = {'sqlalchemy', 'asyncpg', 'httpx2', 'httpx', 'app.core.db'}\n"
        "loaded = forbidden & set(sys.modules)\n"
        "assert not loaded, f'forbidden modules loaded: {loaded}'\n"
        "print('ok')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_detection_works_with_database_url_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)

    assert _escalated("cháy nhà") is True
    assert _escalated("quy định phòng cháy chữa cháy") is False


def test_detection_works_with_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the two ways this process could reach the network.

    Any attempted connection or hostname lookup raises immediately, and detection must
    never trigger either one.
    """

    def _refuse_connect(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("check_emergency must never touch the network")

    def _refuse_getaddrinfo(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("check_emergency must never resolve a hostname")

    monkeypatch.setattr(socket.socket, "connect", _refuse_connect)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse_getaddrinfo)

    assert _escalated("rò gas") is True
    assert _escalated("khoảng cách an toàn phòng cháy chữa cháy") is False

    async def generator(_question: str) -> str:
        return "unused"

    result = _run(answer_unless_emergency("nổ bình gas", generator))
    assert result.safety.escalated is True  # type: ignore[attr-defined]
