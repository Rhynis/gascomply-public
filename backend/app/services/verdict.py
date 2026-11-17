"""Code-side Có/Không verdict check (issue #195).

PR #194's review found that a yes/no answer can open with the wrong verdict while still
citing the exact clause that states the right one: `qwen3:8b`, shown Nghị định 105/2025's
Phụ lục VII mục 15 ("... tồn chứa trên 500 kg"), answered "Có, ... phải mua ..." for a 300 kg
store 5 out of 5 times once the surrounding context changed -- a confident, wrong comparison
on the deciding fact, not a citation problem (the grounding verifier correctly accepts the
citation, since the cited clause really is the one the question is about). A prompt
instruction (`_YES_NO_INSTRUCTION_VI`, issues #173/#175) is a request, not a guarantee, so a
deterministic check is needed, the same reasoning issue #90 already applied to citations.

This module is a direct port of the planner's prototype
(`gascomply-archive/probes/verdict_check_probe.py`), with no rule changed:

1. Find the one quantity ``question`` states: a number plus a unit.
2. Find the one threshold, in the same unit, that a cited clause's own text states with a
   comparison word.
3. Compute whether the quantity satisfies it.
4. Compare the result with ``answer``'s own opening "Có"/"Không".

``check`` returns ``"ok"``, ``"contradiction"``, or ``"n/a"`` when there is no single
comparable quantity/threshold pair -- never a guess (see each function's own docstring for
the fail-toward-``"n/a"`` rules: more than one quantity or threshold, a polarity mismatch, or
no opening verdict at all). ``app.services.generator`` runs this on the cited units' own text,
never an article's recomposed text (an article can state several thresholds, which would
silently never match) -- see that module's own docstring for how ``check``'s result is wired
into a refusal.
"""

import re
from dataclasses import dataclass

UNIT = r"(kg|tấn|m2|m²|m3|m³|m|lít|người)"
NUM = r"(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?)"

COMPARATORS: tuple[tuple[str, str], ...] = (
    (r"không\s+(?:vượt\s+quá|quá|lớn\s+hơn|cao\s+hơn)\s*" + NUM + r"\s*" + UNIT, "<="),
    (r"(?:tối\s+đa|nhiều\s+nhất)(?:\s+là)?\s*" + NUM + r"\s*" + UNIT, "<="),
    (r"không\s+(?:dưới|nhỏ\s+hơn|thấp\s+hơn|ít\s+hơn)\s*" + NUM + r"\s*" + UNIT, ">="),
    (r"(?:tối\s+thiểu|ít\s+nhất)(?:\s+là)?\s*" + NUM + r"\s*" + UNIT, ">="),
    (r"từ\s*" + NUM + r"\s*" + UNIT + r"\s+trở\s+lên", ">="),
    (r"(?:trên|vượt\s+quá|lớn\s+hơn|cao\s+hơn|hơn)\s*" + NUM + r"\s*" + UNIT, ">"),
    (r"(?:dưới|nhỏ\s+hơn|thấp\s+hơn|ít\s+hơn)\s*" + NUM + r"\s*" + UNIT, "<"),
)
"""Order matters: the negated forms come first, and each match is consumed (``thresholds``
below substitutes it out of the text) before a later, looser pattern can re-read the same
span -- so "không vượt quá 70 kg" is read as ``<=`` once, never also as ``vượt quá 70 kg``
(``>``) by a later pattern re-matching what the negated form already covered."""


def _num(text: str) -> float:
    """Parse a Vietnamese-formatted number: "." groups thousands, "," is the decimal point.

    ``"1.000"`` is 1000, not 1 -- the prototype's own first version got this wrong (issue
    #195's own trap), because stripping "." before "," on an input with neither would do
    nothing wrong, but doing it in the other order on "1.000" would read the "." as a decimal
    point and silently truncate to "1". This replaces "." (thousands) first, then "," (decimal).
    """
    return float(text.replace(".", "").replace(",", "."))


def _norm_unit(unit: str) -> str:
    """Fold the two unit spellings that use a superscript digit onto their ASCII form."""
    return {"m²": "m2", "m³": "m3"}.get(unit, unit)


def thresholds(clause: str) -> list[tuple[str, float, str]]:
    """Return every ``(op, limit, unit)`` threshold ``clause``'s own text states.

    Runs every pattern in ``COMPARATORS``' own order over ``clause``, consuming each match
    (substituting it with a space) before the next pattern runs, so a negated form is never
    re-read by the looser pattern it was specifically ordered ahead of.
    """
    text, found = clause.lower(), []
    for pattern, op in COMPARATORS:
        for match in re.finditer(pattern, text):
            found.append((op, _num(match.group(1)), _norm_unit(match.group(2))))
        text = re.sub(pattern, " ", text)
    return found


def quantities(question: str) -> list[tuple[float, str]]:
    """Return every ``(value, unit)`` quantity ``question``'s own text states."""
    return [
        (_num(n), _norm_unit(u))
        for n, u in re.findall(NUM + r"\s*" + UNIT + r"\b", question.lower())
    ]


def satisfied(value: float, op: str, limit: float) -> bool:
    """Return whether ``value`` satisfies the comparison ``op`` against ``limit``."""
    return {">": value > limit, ">=": value >= limit, "<": value < limit, "<=": value <= limit}[op]


def opening_verdict(answer: str) -> str | None:
    """Return ``"Y"``/``"N"`` for an answer opening "Có"/"Không"/"Chưa", else ``None``.

    ``None`` covers "Chỉ khi ...", an echo of the question, or anything else -- there is
    nothing in those openings to compare against a threshold, so ``check`` must give ``"n/a"``
    rather than guess.
    """
    first = answer.strip().lower()
    if re.match(r"^có\b", first):
        return "Y"
    if re.match(r"^(không|chưa)\b", first):
        return "N"
    return None


NEGATIVE_POLARITY = re.compile(
    r"\b(vi\s+phạm|bị\s+phạt|xử\s+phạt|được\s+miễn|miễn|không\s+đủ|không\s+đạt|thiếu|"
    r"không\s+cần|không\s+phải)\b"
)
"""A question asking about the opposite of the clause's own condition ("có vi phạm ...
không?"): the verdict's polarity flips relative to what the clause's threshold states, so
``check`` gives ``"n/a"`` rather than ever guessing which way it flips."""


@dataclass(frozen=True)
class VerdictFacts:
    """The parsed facts behind one ``"contradiction"`` result -- never the question or answer.

    ``app.services.generator`` logs exactly these fields when it refuses (issue #195's own
    trap: the refusal's log line may carry only parsed facts, never a row's question or answer
    prose, since the holdout runner must never be able to echo either back from a log).
    """

    quantity: float
    unit: str
    op: str
    threshold: float
    expected: str
    """``"Có"`` or ``"Không"`` -- what the clause's own threshold implies for ``quantity``."""
    opened: str
    """``"Có"`` or ``"Không"`` -- what ``answer`` actually opened with."""


_VERDICT_VI = {"Y": "Có", "N": "Không"}


def evaluate(
    question: str, clause_texts: list[str], answer: str
) -> tuple[str, VerdictFacts | None]:
    """Run the full check once, returning both ``check``'s own result and its facts.

    The single implementation both ``check`` (the ported, three-way result) and
    ``app.services.generator`` (which additionally needs the facts to log) call -- so there is
    only ever one place the rules below are applied, not two copies that could drift.

    Applies only when ``question`` states exactly one quantity, ``clause_texts`` together
    state exactly one threshold in that quantity's unit, and ``answer`` opens with a verdict
    (``opening_verdict``) -- any other combination is ``"n/a"``, by the same fail-toward-``"n/a"``
    rule ``check`` documents: zero or several quantities, zero or several same-unit
    thresholds, a polarity mismatch (``NEGATIVE_POLARITY``), or no opening verdict at all never
    produce a guess.

    Returns:
        ``("ok", None)`` when the opening verdict matches what the threshold implies;
        ``("contradiction", facts)`` when it does not; ``("n/a", None)`` when nothing here
        applies at all.
    """
    if NEGATIVE_POLARITY.search(re.sub(r"\s*không\s*\??\s*$", "", question.lower())):
        return "n/a", None
    qs = quantities(question)
    ts = [t for c in clause_texts for t in thresholds(c)]
    verdict = opening_verdict(answer)
    if len(qs) != 1 or verdict is None:
        return "n/a", None
    value, unit = qs[0]
    same_unit = {t for t in ts if t[2] == unit}
    if len(same_unit) != 1:
        return "n/a", None
    op, limit, _ = same_unit.pop()
    expected = "Y" if satisfied(value, op, limit) else "N"
    if verdict == expected:
        return "ok", None
    facts = VerdictFacts(
        quantity=value,
        unit=unit,
        op=op,
        threshold=limit,
        expected=_VERDICT_VI[expected],
        opened=_VERDICT_VI[verdict],
    )
    return "contradiction", facts


def check(question: str, clause_texts: list[str], answer: str) -> str:
    """Return ``"ok"``, ``"contradiction"`` or ``"n/a"`` against the cited clause text(s).

    Args:
        question: The question, exactly as asked -- read only for the one quantity it states.
        clause_texts: The cited unit(s)' own text, as shown in the prompt -- never an
            article's whole recomposed text, which usually states several thresholds and
            would make this silently always return ``"n/a"``.
        answer: The model's own answer text -- read only for its opening word.
    """
    result, _ = evaluate(question, clause_texts, answer)
    return result
