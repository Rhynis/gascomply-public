"""Pre-LLM gas-emergency detection (build plan A6 Phase 3.5).

A person who smells gas does not need a citation to Dieu 53; they need to leave the building
and call 114. This module decides, from the raw question alone, whether the question reads as
an emergency happening now, and if so hands back a hard-coded response. It runs before
retrieval and before any LLM call, and it must work when everything else is down, so it uses
the standard library and pydantic only: no database, no network, no model. A test pins that
(``tests/test_safety.py``), so an accidental ``import`` of anything heavier fails CI.

Why this is hard
----------------
The whole corpus is about fire safety. Measured on the 706 ingested chunks, "chay" (fire)
occurs 3,550 times and about 85% of those are "phong chay" or "chua chay" (fire prevention,
fire fighting). The central statute is literally named "Luat Phong chay, chua chay va cuu nan,
cuu ho". A keyword match on "chay" therefore fires on the majority of ordinary questions, and
a warning that cries wolf teaches users to ignore the one that matters. Both errors are
serious, so the detector separates *happening now* from *subject matter*:

1. **Live rules** describe an event that carries its own "now" marker: a perceiving verb
   ("ngui thay mui gas"), an aspect marker ("dang chay", "vua no"), or a sentence-final
   "roi". They are checked on the raw text.
2. **Bare rules** are event vocabulary that the regulations also use ("bi chay",
   "gas ro", "no binh gas"). They are checked only after *subject-matter compounds* have
   been blanked out ("phong chay", "chua chay", "chay no", "kiem tra ro ri", ...), which is
   what keeps "phong chay chua chay nha xuong" from reading as "chay nha".
3. A **rules-question gate** withdraws a match when the question is plainly about the rules:
   it uses rules vocabulary ("quy dinh", "trach nhiem", "xu phat", ...) and the speaker does
   not put themselves in the scene ("toi", "em", "minh"). Bare matches are withdrawn on that
   alone; live matches also need a conditional word ("neu", "khi", "truong hop"), so that
   "Neu ngui thay mui gas thi cua hang phai lam gi theo quy dinh?" is a rules question but
   "Binh gas nha toi dang ro, theo quy dinh toi phai lam sao?" is not.
4. A **negation guard** drops a match directly preceded by "khong", "chua", "het", ...

(The text above is transliterated only to keep this docstring ASCII; the patterns below are
Vietnamese with diacritics, because they have to match Vietnamese.)

Text handling
-------------
Users type with and without diacritics, in NFC or NFD, in any case, with stray punctuation
and repeated letters. ``normalise_question`` makes all of those one canonical text, and
``fold_diacritics`` gives the accent-free view of it. Unaccented text is a real hazard
because "chay" is also "chay" (run), "chay" (flow) and "chay" (vegetarian), so the
unaccented rules apply only when the question carries no diacritic at all (someone typing on
a bare keyboard); a question with any diacritic is judged on its accented text only. Every
unaccented rule is narrower than its accented twin for that reason.

Every judgment call is recorded in ``tests/test_safety.py`` and pinned there, so widening the
net later fails a test instead of happening silently.
"""

import re
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final

from app.schemas.safety import AnswerT, GuardedAnswer, SafetyCheck

EMERGENCY_RESPONSE_VI: Final[str] = (
    "KHẨN CẤP: ra khỏi nơi này ngay và gọi 114.\n"
    "\n"
    "1. Ra khỏi nhà hoặc khu vực ngay.\n"
    "2. Nếu còn an toàn, đóng van bình gas.\n"
    "3. Không dùng lửa, không bật hoặc tắt công tắc điện.\n"
    "4. Gọi 114 (Cảnh sát PCCC và CNCH)."
)
"""The hard-coded emergency copy (user-facing product text, so Vietnamese: C8).

Written for someone holding a phone in a panic: the first line alone says leave and call 114,
and the four numbered actions repeat it in order. It carries exactly four actions (get out,
close the valve if still safe, no flames and no electrical switches, call 114) and no other
safety advice: wrong or incomplete advice is worse than none. Returned as this one object on
every escalation, so the bytes never vary; ``tests/test_safety.py`` pins them by hash.
"""

# --- text normalisation ----------------------------------------------------------------

_NOT_WORD: Final = re.compile(r"[\W_]+")
"""Anything that is not a letter or a digit. Runs after NFKC composition, so a Vietnamese
letter is always one precomposed character and is never split by this."""

_ELONGATION: Final = re.compile(r"(\w)\1{2,}")
"""Three or more of the same character in a row ("chayyyy"). Vietnamese words never contain
that, so it is safe to collapse to one; two in a row is left alone ("gass" is a plain typo
handled by the patterns)."""


def normalise_question(question: str) -> str:
    """Return the canonical text every rule runs on.

    Composes NFC/NFD and full-width forms to one shape (NFKC), lower-cases (``Đ`` becomes
    ``đ``), turns punctuation and symbols into spaces, collapses ``chayyyy`` style
    elongation, and squeezes whitespace. Diacritics are kept; see ``fold_diacritics``.

    Args:
        question: The raw text the user typed.

    Returns:
        Lower-case words separated by single spaces.
    """
    composed = unicodedata.normalize("NFKC", question).lower()
    spaced = _NOT_WORD.sub(" ", composed)
    return " ".join(_ELONGATION.sub(r"\1", spaced).split())


def fold_diacritics(text: str) -> str:
    """Return ``text`` with every diacritic removed and ``đ`` written as ``d``.

    Args:
        text: Text, ideally already passed through ``normalise_question``.

    Returns:
        The accent-free spelling, e.g. ``cháy nhà`` becomes ``chay nha``.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return stripped.replace("đ", "d").replace("Đ", "D")


# --- vocabulary, accented and unaccented ------------------------------------------------
#
# Rule sources are written once, in Vietnamese with diacritics, using %TOKEN% placeholders
# for the few vocabularies whose unaccented form needs a different (narrower) spelling than
# ``fold_diacritics`` would give. Everything else in a source is folded automatically for the
# unaccented set.

_ACCENTED_TOKENS: Final = {
    # How a person spells gas. "ga" is common; "lpg" is the regulations' word and is kept out
    # of the bare rules on purpose (a person in a panic says "gas", the clause says "LPG").
    "%GAS%": r"(?:gass?|gaz|ga)",
    "%GASANY%": r"(?:gass?|gaz|lpg|ga)",
    # A leak. "rỉ" alone also means rust ("rỉ sét"). "rỏ" (to drip) is a one-tone-mark typo
    # for "rò" that a panicking typist plausibly makes; the corpus never uses "rỏ".
    "%LEAK%": r"(?:rò rỉ|rò|rỏ|xì|hở|rỉ(?! sét))",
    "%LEAKING%": r"(?:rò rỉ|rò|rỏ|xì|rỉ(?! sét))",
    # Somewhere a fire can be (the speaker's own place). No "phòng": "phòng cháy" is the
    # fire-prevention compound, not a room on fire.
    "%PLACE%": r"(?:nhà|cửa hàng gas|cửa hàng ga|cửa hàng|quán|tiệm|kho|xưởng|bếp|chợ|căn hộ"
    r"|chung cư|xe|bình gas|bình ga|bếp gas|bếp ga|kho gas|kho ga)",
    # The speaker putting themselves in the scene.
    "%PERSONAL%": r"(?:tôi|tui|tớ|em|mình|cháu)",
}

_UNACCENTED_TOKENS: Final = {
    # "xi ga" is "xì gà" (a cigar) once accents are gone; "xi gas" stays a gas hiss.
    "%GAS%": r"(?:gass?|gaz|(?<!xi )ga)",
    "%GASANY%": r"(?:gass?|gaz|lpg|(?<!xi )ga)",
    "%LEAK%": r"(?:ro ri|ro|xi|(?:ri(?! set)))",
    "%LEAKING%": r"(?:ro ri|ro|xi|(?:ri(?! set)))",
    # Left out on purpose because they collide with everyday words once accents are gone:
    # "xe chay" (xe chạy), "quan chay" (quán chay), "tiem chay", "cho chay" (chợ cháy vs
    # cho chạy), "cua hang chay" (cửa hàng chay, a vegetarian shop), "bep chay", and "kho
    # chay" (kho cháy vs the regulations' "khó cháy", hard to burn).
    "%PLACE%": r"(?:nha|xuong|chung cu|can ho|binh gas|binh ga|bep gas|bep ga|kho gas|kho ga"
    r"|cua hang gas|cua hang ga)",
    # "to" (tớ), "chau" (cháu) and "toi" as "tới" are too common as ordinary words unaccented;
    # they were the reason the rules-question gate opened by mistake on the folded corpus.
    "%PERSONAL%": r"(?:toi|tui|em|minh)",
}

_SEPARATOR: Final = " ¦ "
"""Replaces a blanked-out subject-matter compound. Not a word character, so it also breaks
adjacency: "phòng cháy nhà ở" must not leave "cháy nhà" behind."""


@dataclass(frozen=True)
class _RuleSpec:
    """One detection rule as written: a name, an accented pattern, and where it applies."""

    name: str
    source: str
    unaccented: bool = True
    """Whether the rule also runs on accent-free text. False when its unaccented spelling
    collides with ordinary words ("no roi" is also "no rồi", full up)."""


# Live rules: the phrase carries its own "now" marker, so it does not need help to be read as
# an event happening at the speaker's place.
_LIVE_SPECS: Final = (
    # A smell of gas, perceived or present, with the literal word "mùi": "ngửi thấy mùi
    # gas", "nhà có mùi gas", "mùi gas nồng nặc". Kept unaccented (folds to "mui", which is
    # not overloaded the way "hơi"/"hôi" both fold to "hoi" -- see gas_smell_secondary).
    _RuleSpec(
        "gas_smell",
        r"\b(?:ngửi|nghe|thấy|có|bốc|sặc|nồng)(?: thấy| được)? mùi (?:khí )?%GASANY%\b"
        r"|\bmùi (?:khí )?%GASANY% (?:nồng|nặc|nặng|hôi|bốc|xộc|lan|khắp|rất|quá|khá|lắm)\b",
    ),
    # The same smell, said without "mùi": "hôi gas", "sặc gas", "nồng nặc gas". Accented only:
    # "hôi" (stinky) and "hơi" (vapor, the corpus's own word for LPG's gas phase) both fold to
    # "hoi", so unaccented this fires on ordinary sentences like "hơi LPG" (measured on the
    # corpus: "phân tán hơi LPG", "van hơi ... hơi LPG").
    _RuleSpec(
        "gas_smell_secondary",
        r"\b(?:hôi|sặc|nồng|nồng nặc) (?:khí )?%GASANY%\b",
        unaccented=False,
    ),
    # A leak in progress: "gas đang phun ra", "gas xì ra ngoài". Not "đang/vừa + %LEAKING%"
    # unaccented: "đang" (progressive marker) folds to "dang", which collides with "dạng"
    # (shape), so it fires on "biến dạng, rò rỉ" (a QCVN test-defect list, not a live leak).
    # The bare "gas_leak" rule below already covers "gas đang rò" without that ambiguity.
    _RuleSpec(
        "leak_now",
        r"\b(?:đang|vừa) (?:bị )?%LEAKING%\b"
        r"|\b%GAS% (?:đang |vừa )?(?:phun|tuôn|tràn|xì|thoát|rỉ|rò) ra\b",
        unaccented=False,
    ),
    # A fire in progress: "đang cháy", "vừa xảy ra cháy", "cháy rồi", "cháy dữ dội".
    _RuleSpec(
        "fire_now",
        r"\b(?:đang|vừa) (?:bị |xảy ra |có |bốc |bùng |bùng phát )?cháy\b"
        r"|\bcháy (?:rồi|quá|lắm|ngùn ngụt|dữ dội)\b"
        r"|\b(?:vừa|mới) (?:xảy ra|có) (?:một |1 )?(?:vụ |đám )?cháy\b",
        unaccented=False,
    ),
    # An explosion that has just happened: "vừa nổ", "nổ rồi", "vừa nghe tiếng nổ", "vừa xảy
    # ra vụ nổ" (mirrors fire_now's "vừa xảy ra vụ cháy").
    _RuleSpec(
        "explosion_now",
        r"\b(?:đang|vừa) (?:bị )?nổ\b"
        r"|\bnổ rồi\b"
        r"|\b(?:nghe|có|thấy|vừa có|vừa nghe|vừa thấy)(?: thấy)? (?:một |1 )?tiếng nổ\b"
        r"|\b(?:vừa|mới) (?:xảy ra|có) (?:một |1 )?vụ nổ\b",
        unaccented=False,
    ),
    # A gas leak heard rather than smelled: "nghe tiếng xì xì", "có tiếng xì từ bình gas". The
    # sound word "xì xì" (a hiss) does not appear anywhere in the corpus's compliance text.
    _RuleSpec(
        "leak_sound",
        r"\b(?:nghe|có|thấy) tiếng xì(?: xì)?\b",
        unaccented=False,
    ),
)

# Bare rules: event vocabulary that the regulations also use. They run on text with the
# subject-matter compounds blanked out, and only fire when the rules-question gate is shut.
_BARE_SPECS: Final = (
    # "rò gas", "xì gas", "gas bị rò", "bình gas bị rò", "bình bị rò", "van bị hở".
    _RuleSpec(
        "gas_leak",
        r"\b%LEAK% (?:khí )?%GAS%\b"
        r"|\b%GAS% (?:đang |bị |vừa )?%LEAK%\b"
        r"|\b(?:bình|bếp|van|dây)(?: đang| bị)? %LEAK%\b",
    ),
    # A fire at a place, or in a gas appliance: "cháy nhà", "nhà bị cháy", "cửa hàng cháy",
    # "bình gas bốc cháy", "cháy to".
    _RuleSpec(
        "fire",
        r"\b%PLACE%(?: (?:tôi|em|mình|bên|kế bên|bên cạnh|hàng xóm))?(?: đang| bị| vừa| đã)? cháy\b"
        r"|\bcháy %PLACE%\b"
        r"|\b(?:bốc|bùng|bén) (?:cháy|lửa)\b"
        r"|\bcháy (?:to|lớn)\b",
    ),
    # "Lửa bùng lên" (flame flaring up), subject-first. Accented only: "lửa bén" folds to "lua
    # ben", colliding with "bên" (side) -- the corpus has "nguồn lửa bên ngoài" (an ignition
    # source outside), which "lua ben" would wrongly match. "bùng"/"bốc" alone do not collide
    # (checked against the corpus), but are kept in this narrower, accented-only rule anyway
    # for the same reason: this alternative reads subject before verb, which is not how the
    # corpus's own "bốc/bùng cháy" phrasing above already unaccented-tests clean.
    _RuleSpec(
        "fire_flame",
        r"\blửa (?:bùng|bốc)(?: lên)?\b",
        unaccented=False,
    ),
    # A gas cylinder or stove exploding: "nổ bình gas", "bình gas nổ", "bình gas phát nổ",
    # "binh gas no roi" (unaccented). "nổ" folds to "no", an ordinary word ("no" = full from
    # eating), so this stays scoped to "nổ" directly touching gas/bình/chai/bếp -- narrow
    # enough that the corpus sweep (706 chunks) has zero hits in that shape.
    _RuleSpec(
        "explosion",
        r"\b(?:nổ|phát nổ|bị nổ) (?:bình|chai) %GAS%\b"
        r"|\b(?:bình|chai|bếp) %GAS% (?:bị |vừa |đã |phát )?nổ\b"
        r"|\b%GAS% (?:bị |vừa |đã |phát )?nổ\b"
        r"|\bbình (?:bị |vừa |đã |phát )nổ\b",
    ),
)

# Subject-matter compounds, blanked out before the bare rules run. Built from the corpus:
# the words that precede "cháy" and "nổ" most often are phòng, chữa, báo, ngăn, chống,
# nguy hiểm, vụ, dễ, gây (see the PR for the counts). "phát hiện" is deliberately absent from
# the leak compounds: "tôi phát hiện rò gas" is a report, "thiết bị phát hiện rò rỉ" is not.
_MASK_SOURCE: Final = (
    r"\bcháy nổ\b"
    r"|\b(?:phòng|chữa|chống|ngăn|báo|dập|tránh|ngừa|đề phòng|bảo hiểm|nguy hiểm|nguy cơ|"
    r"nguồn gây|gây|dễ|tự|vụ|đám|do|điểm|nhiệt độ|khả năng|vật liệu|chất|khoang|vết|thử|"
    r"áp suất|thuốc|liệu|an toàn|kiểm soát|hạn chế)(?: ngừa)? (?:bắt |bốc )?(?:cháy|nổ)\b"
    r"|\b(?:kiểm tra|thử|dò|cảnh báo|báo|đo|chống|ngăn|phòng|tránh|ngừa|hạn chế|kiểm soát|"
    r"cảm biến|đầu dò|thiết bị dò|máy dò|xử lý|khắc phục)(?: ngừa)? (?:sự cố )?%LEAK%\b"
)

# The rules-question gate. Vocabulary of someone asking about the rules or about the causes
# of an incident, not reporting one.
_FRAME_SOURCE: Final = (
    r"\b(?:quy định|quy chuẩn|tiêu chuẩn|pháp luật|luật|nghị định|thông tư|nghĩa vụ|trách nhiệm"
    r"|xử phạt|mức phạt|phạt|thẩm định|thẩm duyệt|giấy phép|giấy chứng nhận|hồ sơ|thủ tục"
    r"|điều kiện|nguyên nhân|tại sao|vì sao|là gì|có được|được phép|không được|có phải|có cần"
    r"|bắt buộc|phòng ngừa|phòng tránh|ngăn ngừa|biện pháp|yêu cầu|điều \d+|khoản \d+)\b"
)
_PERSONAL_SOURCE: Final = r"\b%PERSONAL%\b"
_CONDITIONAL_SOURCE: Final = r"\b(?:nếu|giả sử|giả dụ|trường hợp|khi|lỡ|nhỡ|hễ)\b"
_NEGATORS: Final = ("không", "chưa", "chẳng", "hết", "đừng", "đâu")

_SHOUT_FILLER: Final = frozenset(
    {"rồi", "to", "lớn", "quá", "lắm", "nhà", "cứu", "giúp", "với", "ơi", "ạ", "tôi", "em", "có"}
)


@dataclass(frozen=True)
class _Rule:
    """A compiled rule."""

    name: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class _PatternSet:
    """Everything needed to judge one view of the text (accented or unaccented)."""

    live: tuple[_Rule, ...]
    bare: tuple[_Rule, ...]
    mask: re.Pattern[str]
    frame: re.Pattern[str]
    personal: re.Pattern[str]
    conditional: re.Pattern[str]
    negators: frozenset[str]
    shout: bool


def _compile(source: str, tokens: dict[str, str], *, fold: bool) -> re.Pattern[str]:
    """Substitute the vocabulary tokens, fold the rest when unaccented, and compile."""
    text = fold_diacritics(source) if fold else source
    for token, spelling in tokens.items():
        text = text.replace(token, spelling)
    return re.compile(text)


def _build(*, fold: bool) -> _PatternSet:
    tokens = _UNACCENTED_TOKENS if fold else _ACCENTED_TOKENS

    def rules(specs: tuple[_RuleSpec, ...]) -> tuple[_Rule, ...]:
        return tuple(
            _Rule(spec.name, _compile(spec.source, tokens, fold=fold))
            for spec in specs
            if spec.unaccented or not fold
        )

    return _PatternSet(
        live=rules(_LIVE_SPECS),
        bare=rules(_BARE_SPECS),
        mask=_compile(_MASK_SOURCE, tokens, fold=fold),
        frame=_compile(_FRAME_SOURCE, tokens, fold=fold),
        personal=_compile(_PERSONAL_SOURCE, tokens, fold=fold),
        conditional=_compile(_CONDITIONAL_SOURCE, tokens, fold=fold),
        negators=frozenset(fold_diacritics(word) if fold else word for word in _NEGATORS),
        shout=not fold,
    )


_ACCENTED: Final = _build(fold=False)
_UNACCENTED: Final = _build(fold=True)


def _negated(text: str, start: int, negators: frozenset[str]) -> bool:
    """True when one of the two words before ``start`` negates what follows."""
    return any(word in negators for word in text[:start].split()[-2:])


def _first_hit(rules: tuple[_Rule, ...], text: str, negators: frozenset[str]) -> str | None:
    """Name of the first rule with a match that is not directly negated."""
    for rule in rules:
        for match in rule.pattern.finditer(text):
            if not _negated(text, match.start(), negators):
                return rule.name
    return None


def _is_shout(text: str) -> bool:
    """True for a one-word alarm: "cháy", "cháy rồi", "cứu với cháy" (at most four words)."""
    words = text.split()
    return (
        "cháy" in words
        and len(words) <= 4
        and all(word == "cháy" or word in _SHOUT_FILLER for word in words)
    )


def _judge(text: str, patterns: _PatternSet) -> str | None:
    """Name of the rule that fires on ``text`` under ``patterns``, or ``None``."""
    rules_question = bool(patterns.frame.search(text)) and not patterns.personal.search(text)
    hedged = rules_question and bool(patterns.conditional.search(text))
    live = _first_hit(patterns.live, text, patterns.negators)
    if live is not None and not hedged:
        return live
    if rules_question:
        return None
    bare = _first_hit(patterns.bare, patterns.mask.sub(_SEPARATOR, text), patterns.negators)
    if bare is not None:
        return bare
    return "shout" if patterns.shout and _is_shout(text) else None


def check_emergency(question: str) -> SafetyCheck:
    """Decide whether ``question`` reads as a gas emergency happening now.

    Pure, synchronous and offline: no database, no network, no model. Safe to call before
    anything else in ``/ask``.

    Args:
        question: The raw question exactly as the user typed it.

    Returns:
        ``SafetyCheck(escalated=True, ...)`` carrying ``EMERGENCY_RESPONSE_VI`` when it is an
        emergency, otherwise ``SafetyCheck(escalated=False)``.
    """
    text = normalise_question(question)
    matched = _judge(text, _ACCENTED)
    if matched is None and fold_diacritics(text) == text:
        # No diacritic anywhere: the user is on a bare keyboard, so the unaccented rules apply.
        unaccented = _judge(text, _UNACCENTED)
        matched = None if unaccented is None else f"{unaccented}/unaccented"
    if matched is None:
        return SafetyCheck(escalated=False)
    return SafetyCheck(escalated=True, matched=matched, response=EMERGENCY_RESPONSE_VI)


async def answer_unless_emergency(
    question: str, generate: Callable[[str], Awaitable[AnswerT]]
) -> GuardedAnswer[AnswerT]:
    """Run ``generate`` for ``question`` unless the question is a gas emergency.

    This is the seam ``/ask`` (Phase 3.7) calls. ``generate`` is the whole downstream
    pipeline (retrieval and the LLM). On the emergency path it is never invoked, not even to
    build a coroutine, so no retrieval and no model call can happen.

    Args:
        question: The raw question exactly as the user typed it.
        generate: Async callable that answers a question; only called for a non-emergency.

    Returns:
        A ``GuardedAnswer`` whose ``safety.escalated`` tells the two paths apart; ``answer``
        is ``None`` on the emergency path and ``generate``'s result otherwise.
    """
    safety = check_emergency(question)
    if safety.escalated:
        return GuardedAnswer(safety=safety)
    return GuardedAnswer(safety=safety, answer=await generate(question))
