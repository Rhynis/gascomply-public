"""Generation with enforced citations (build plan A6 Phase 3.6, issue #90).

A model that invents "Điều 61 khoản 3" in the right format is more dangerous than
one that refuses: the answer looks verifiable and is not. So citations are never
taken on trust:

1. The model is asked for JSON matching a fixed schema (``_JSON_SCHEMA``), passed to
   the provider as structured-output ``format`` when it supports one
   (``OllamaProvider``). The reply is parsed through that schema
   (``pydantic.BaseModel.model_validate``) -- never scraped as free text for an
   article number.
2. **Every citation the model gives is checked against the retrieved context.** A
   citation naming a unit that was not actually retrieved for this question is a
   fabrication: it is dropped (``_validate_citations``), and logged loudly, so a
   silent gap in the corpus never becomes a silently-wrong citation in the answer.
   Before that check runs, ``_split_bundled_citations`` recovers a local model
   quirk noted since #107: several real ids sometimes arrive packed into one
   ``unit_id`` string (e.g. ``"...#khoan-2, ...#khoan-3"``) instead of one
   citation object each. Splitting is purely mechanical (ids never contain a
   comma, A5.1's own scheme) and every resulting piece still goes through the
   exact same fabrication check -- this never trusts anything the split step
   produces, it only stops a comma from hiding an otherwise-valid id from
   ``_validate_citations`` entirely.
3. If nothing survives -- no supporting context at all, the model declined, or
   every citation it gave was fabricated -- the result is a **refusal**, a
   first-class ``AskOutcome`` with its own reason code, never an apologetic
   paragraph inside ``answer``.

Retrieval config for generation
--------------------------------
``RetrievalConfig.passage_limit`` defaults to 10 (``app/schemas/retrieval.py``), and
a real question already saturates it ("chai LPG phải kiểm định định kỳ thế nào"
returns exactly 10 passages against the reviewed corpus). An answer that legitimately
spans several clauses plus the obligations they state needs headroom above that
default, not silent truncation, so generation uses its own
``GENERATION_RETRIEVAL_CONFIG`` (``passage_limit=15``, ``entity_limit=12``) rather
than the retriever's bare default: fifteen because the module's own reproduction of
the issue's hand-built example (Điều 53's five clauses, each a focus unit inside one
chunk, plus the articles they cite) fits well inside it with headroom for a second
seed hit; twelve entities because the largest observed obligation set on the reviewed
corpus (a concept plus its 8 requirements) fits with room to grow before the corpus
does.

``k`` -- /ask's own seed width, chosen by measurement (issue #169)
-----------------------------------------------------------------------
``RetrievalConfig.k`` (how many vector hits seed graph expansion) was never set here at
all before this issue -- it silently fell through to ``RetrievalConfig``'s own bare
default of 5, which nobody had actually chosen for generation (every recall@k number
since #96 is measured at ``k=10``; PR #168's review is what first noticed the gap). This
issue swept 4 points (``eval.compare --generation-k``/``--generation-passage-limit``, one
full run each, qwen3:8b, dev DB, read-only) and chose among them by blocking rows first,
then strict citation accuracy, then effective-answer rate -- never by raising a fifth
point to rescue a row that failed:

=====  ====  ======  ======  ======  ==========  ==========
point  k     p_lim   recall  cit.acc eff.answer  station-el.
=====  ====  ======  ======  ======  ==========  ==========
A      5     15      0.864   0.753   0.939 PASS  FAIL (0.000)
B      8     15      0.864   0.769   0.788 FAIL  FAIL (refuses)
C      10    15      0.864   0.821   0.848 FAIL  PASS (1.000)
D      10    20      0.864   0.815   0.818 FAIL  PASS (1.000)
=====  ====  ======  ======  ======  ==========  ==========

Every point's flagship question (``multihop-gas-store-fire-insurance``, gold unit
``pccc-decree-105-2025#phu-luc-7#muc-15``): A answers and cites the gold unit (wrong
*conclusion* -- the next issue's own job; see "Checking the deciding fact" above); B, C
and D all refuse it outright -- nothing to cite at all.

Point A already fails the station-electrical blocking row (it answers, but cites the
wrong unit: 0/1 precision) -- that is this issue's own reason to exist, and why A is the
baseline, not a candidate. Raising ``k`` to 8 or 10 does not trade that failure for a
clean pass: B outright fails effective-answer rate (several multi-hop rows that *answer*
at A start *refusing* at wider k -- more seed hits bring in more distractor passages for
the per-citation verifier to reject, not more signal) and still fails station-electrical
by refusing it entirely rather than answering wrong. C and D do fix station-electrical
(the wider seed set finally pulls ``lpg-station-qcvn-10-2012#dieu-14`` into the pool), but
both then take the flagship question from "answers, wrong conclusion" to "refuses
outright" (see above) -- strictly worse on the one metric that matters most here. No
swept point passes every blocking row, which the issue's own decision rule accepts as a
valid outcome: keep A, state ``k`` explicitly rather than leaving it to
``RetrievalConfig``'s bare default, and record why here rather than silently re-choosing
5 every time someone reads the default.

Marking each focus unit its own citable line, not a bare id list (issue #108)
-------------------------------------------------------------------------------
Citation *accuracy* (precision: of the ids the model actually cited, how many are
right) stayed at 0.45-0.56 well after #107 fixed citation *recall* -- and measuring
every answerable gold row (not the handful #108 first sampled) against what
``Retriever.retrieve`` actually returned showed the gold unit was already sitting in
the model's own context, citable, in most of the failing rows. The context just
never let the model tell it apart from its siblings: ``_build_user_prompt`` wrote a
passage's own (usually whole-article) id as a proper ``[unit_id: ...]`` line, but
every finer-grained ``focus_unit`` #107 made citable was listed underneath as a bare,
unlabelled id in one comma-joined Vietnamese aside -- not in the shape
``_SYSTEM_PROMPT_VI`` rule 2 itself says is the *only* valid citation source ("CHÍNH
XÁC từ các dòng '[unit_id: ...]'"). A model that took that rule at its word had every
reason to fall back to the coarse passage id it was actually shown as a marked line,
even when its answer really rested on one specific Khoản or Điểm inside it -- which
exact-match citation accuracy scores as wrong, unlike containment recall.

The fix is only ever a presentation change: each focus unit now gets its own
``[unit_id: ...]`` line (with its own Điều/Khoản/Điểm label) directly under its
passage's text, so rule 2's own wording already covers it -- no new exception, no
loosened validation, nothing added to what ``_build_sources`` already considered
citable. A new rule 6 asks the model to prefer the most specific matching id over the
whole-passage one and never to cite every sibling at once, addressing the mirror
failure (citing several/all of them) the same data showed on multi-hop rows.

Keeping the ``[unit_id: ...]`` marker out of ``answer`` (issue #94)
---------------------------------------------------------------------
``_build_user_prompt`` writes a ``[unit_id: ...]`` marker in front of every citable
unit so the model can name its sources precisely -- that mechanism stays, it is how
a citation is checkable at all. Left alone, though, a model sometimes echoes the
marker (or a lookalike of it) into the prose of ``answer`` itself, e.g. "... Điều 53
Nghị định 87/2018/NĐ-CP (unit_id: gas-decree-87-2018#dieu-53):" -- a raw primary key
leaking into the one field a user actually reads, right next to the ``citations``
array that already carries the same fact in structured form.

Both defenses are applied, not one instead of the other:

1. ``_SYSTEM_PROMPT_VI`` now tells the model outright not to repeat a unit_id inside
   ``answer``. Cheap, sometimes sufficient, but -- exactly as with the citation
   fabrication problem #90 solved -- a prompt instruction is a request, not a
   guarantee, so it cannot be the only defense.
2. ``_strip_leaked_unit_id_markers`` deterministically removes anything shaped like
   the marker from ``parsed.answer`` before a response is built, regardless of
   whether the model obeyed rule 1. It matches only the marker's own shape: the
   literal token ``unit_id:`` (which never occurs in ordinary Vietnamese prose)
   followed by an id in this project's own scheme (``app/core/ids.py``'s
   ``#``/``:``-separated segments), with or without a surrounding bracket. It never
   touches ``citations`` -- only the free-text ``answer`` string is passed through
   it, after validation has already built ``citations`` from ``sources``.

Quoting a cited appendix item verbatim (issue #146)
-------------------------------------------------------
A 7-8B model asked about an obligation that only applies above a threshold (Phụ lục
VII mục 15: "... tồn chứa trên 500 kg") reliably states the obligation but drops the
threshold -- prompting alone did not fix this reliably, and the fix is not another
model call (one more place for a hallucination to hide) but deterministic
post-processing: ``_quote_cited_appendix_items`` appends the exact, stored body of
each cited appendix item or sub-item (``"appendix_item"``/``"appendix_subitem"``,
never a form, an article, a clause, an appendix's own Roman-numbered prose section,
or a lettered point on its own) that is at most 300 characters long, labelled with
its Roman numeral and document number -- both read from data (``AppendixItemRow``'s
own stored coordinates and the citation's own document row), never from the model's
text. It runs last, after both the fabrication check and the grounding check
(including #158's status guard already refused what it refused), over whatever
``citations`` finally survived -- so a quote is only ever appended for a unit the
answer is already allowed to cite, never a reason to keep or drop one.

Offering a Phụ lục VI row's own deductible rule as its own citable passage (issue #165)
-----------------------------------------------------------------------------------------
A Phụ lục VI premium-table row names a deductible type with a bare letter cell
("... N 0,3") but never states that type's actual rule on the row itself -- the real
cap (1% for M, 10% for N) lives on a separate điểm of the appendix's own Mục II khoản
1. A fix round on #166 first tried appending that rule's text straight into the row's
own *chunk* at embedding time; that regressed two unrelated, previously-correct
answers (diluting every row's embedding with the same ~250-character rule text pushed
the flagship and premium questions' real evidence out of the retrieved top-k) and was
reverted -- ``ingest/chunk.py`` is back to byte-identical with `main`.

The fix instead runs at *generation* time, after retrieval, never touching a chunk or
an embedding: ``_with_deductible_rule_passages`` scans the passages already retrieved
for this question, and for every one that is a Phụ lục VI row or sub-row whose own
text carries an "M"/"N" cell, looks up that letter's rule unit
(``DocumentRepository.get_deductible_rule_units``, found by the unit's own text --
never a formula on the id scheme) and links it as a ``focus_units`` entry of that row
(issue #108's own mechanism for a finer-grained citable sub-unit), with a clear,
value-bearing display label (``_deductible_rule_label``, threaded through
``_build_user_prompt`` as ``deductible_rule_labels``) -- the row's own passage *text*
is never touched. Two earlier shapes of this same generation-time idea were tried and
measured against the real model first: a free-text pointer line appended to the row's
text fixed the deductible question but regressed the premium one every time (several
phrasings tried); adding the resolved rule unit as a brand-new standalone ``Passage``
*in addition to* the focus-units link intermittently regressed the flagship question
instead (an emergent interaction specific to that combination). ``_with_deductible_rule_passages``'s
own docstring has the full measured detail; this version -- focus-units link plus
label, nothing else -- was the one that held 8-12/8-12 clean runs on every one of the
flagship, premium and both deductible-wording questions.

This changes nothing about what counts as a valid citation: the grounding verifier
(``_run_grounding``, ``DocumentRepository.get_unit_contexts``, ``_validate_citations``)
is completely unchanged, and a 10% claim is only ever grounded by actually citing the
điểm that states it -- never by "mức 17 supports 10%" through some side channel. The
rule unit being offered as a focus unit, with its own label, is exactly what lets the
model cite it directly, which is the only way #166's original "10% but refused" gap
closes without weakening what a citation means.

Answering a yes/no question with Có/Không, not an echo (issue #173)
---------------------------------------------------------------------
``_is_yes_no_question`` is a pure, deterministic check on the question text: it matches
a Vietnamese yes/no question (a "có"/"còn" particle anywhere, ending in "không"/"ko",
optionally followed by one politeness/tag particle -- "ạ", "vậy" or "nhỉ" -- and
optionally by a "?"), never a "bao nhiêu"/"nào"/"gì"/"thế nào" open question, which never
ends that way. Only when it matches, ``_build_user_prompt`` adds
``_YES_NO_INSTRUCTION_VI`` right before the "CÂU HỎI:" line -- asking the model to open
with "Có"/"Không"/"Chỉ khi ...", state the cited clause's own condition or threshold in
the answer's own words, compare any number the question gives against that threshold
using the clause's own comparison word (never a hardcoded "equal or above"), and never
restate the question as the answer. Every other question's prompt is byte-for-byte
unchanged: the instruction block is never added, added text, or reordered for a
non-match (``tests/test_generator.py`` proves this on the premium question, a real
non-yes/no gold row, against a captured pre-#173 prompt).

``_YES_NO_QUESTION_PATTERN``'s own docstring has this pattern's full history: a widening
tried once (PR #175), measured unsafe and withdrawn, then re-applied as it stands now
(issue #198) once the next section's code-side verdict check gave a backstop the
withdrawal no longer needed to rely on the prompt alone for. ``_YES_NO_INSTRUCTION_VI``
and its comparison-word rewrite (also from PR #175) are unaffected by either change. See
the PR bodies for both issues' full measurements.

Refusing when the answer's own verdict contradicts its own cited clause (issue #195)
-----------------------------------------------------------------------------------------
``_YES_NO_INSTRUCTION_VI`` tells the model how to compare a question's quantity against a
cited clause's threshold; it does not make the model do it. PR #194's review measured
``qwen3:8b`` citing Nghị định 105/2025's Phụ lục VII mục 15 ("... tồn chứa trên 500 kg")
while opening "Có, ... phải mua ..." for a 300 kg store, 5 out of 5 times, once the
surrounding retrieved context changed -- a confident, wrong comparison on the deciding fact,
not a citation problem, since the cited clause really is the right one.

``app.services.verdict`` is a pure, deterministic port of the planner's prototype
(``gascomply-archive/probes/verdict_check_probe.py``), with no rule changed: it finds the
one quantity the question states, the one same-unit threshold a cited clause's own text
states, computes whether the quantity satisfies it, and compares that with the answer's own
opening "Có"/"Không". ``generate_answer`` runs it right after the grounding filter settles
``citations`` (whether or not the grounding check itself is enabled), on each kept citation's
own unit text -- ``_Source.body``, the same per-unit line issue #188 renders in the prompt --
never an article's recomposed text (``documents.get_unit_contexts``'s own per-citation
lookup), which usually states several thresholds and would make the check silently never
fire at all.

``verdict.evaluate`` itself fails toward ``"n/a"`` (no check at all) whenever more than one
quantity or threshold is in play, the question asks about the opposite of the clause's
condition (a polarity guard), or the answer does not open with a verdict at all -- so this
runs unconditionally, on every answer, not only a detected yes/no question: it is already a
no-op everywhere else. Only a ``"contradiction"`` result changes the outcome, to a refusal
with the new reason code ``"verdict_contradicts_clause"`` -- the fail-closed choice, since
the answer's own stated reasoning argues the wrong way and there is no safe way to rewrite
it. The refusal's own log line (``_log_verdict_contradiction``) carries only the parsed
quantity, threshold and the two verdicts -- never the question or answer text, so the
holdout runner can never echo a row's own prose back out of a log.

This check is what makes widening ``_YES_NO_QUESTION_PATTERN`` again (issue #198) safe:
a wrong "Có"/"Không" that now reaches more question endings than before this module
existed is caught right here and refused, rather than shipped as a confident wrong
answer the way PR #175's own withdrawn widening once was, before this check existed.

Retrying once without the passages the grounding verifier rejected (issue #202)
-------------------------------------------------------------------------------------
A right answer can still cite the wrong unit. The planner measured ``qwen3:8b`` citing
Phụ lục VII mục 17 (an unrelated hotel item) for a correct "Có, ... phải mua ..." about a
600 kg gas store, 8 times out of 8, because mục 17's own sibling under a different
appendix (Phụ lục VI's premium table) sits right next to the real answer, mục 15, in the
retrieved context -- the model builds a real unit id out of the wrong appendix and the
wrong mục. The per-citation verifier (layer 2, ``filter_supported_citations``) correctly
rejects that citation; with nothing left to cite, the answer becomes an
``unsupported_citations`` refusal -- safe, but wrong, since the right unit was sitting in
the same context the whole time, just never cited.

``generate_answer`` retries exactly once when that happens: when layer 2 rejects every
citation of the first attempt, and at least one rejected id is the own ``unit_id`` of a
passage that was actually offered (never a focus unit alone, and never a citation a
layer-1 guard refused first -- see ``_retry_without_rejected_passages``'s own docstring
for the exact condition). The retry removes only those passages and runs every later step
unchanged: the same prompt builder, the same fabrication check, the same grounding and
verdict checks, over whatever remains. Nothing is added back -- a retry only ever offers
*less* context than the first attempt, never different or additional passages, so a model
that fabricated once is not simply given a second, wider chance to fabricate again.

A retry is never retried: if the second attempt also refuses, that refusal is final,
whatever its reason. The planner's own simulation (``retry_without_rejected_probe.py``)
measured two retries unsafe on a different row in the same family
(``multihop-gas-store-800kg-insurance``): the first attempt's wrong "Không" is correctly
refused, but a second retry reaches a wrong "Không" answer 3 times out of 3, caught only
by issue #195's verdict check -- which does not exist for every question. One retry keeps
the fix narrow to the failure it was measured against; a second retry resamples a wrong
answer, which this issue does not allow.
"""

import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.config import get_settings
from app.core.ids import ENTITY_SEPARATOR, UNIT_SEPARATOR
from app.llm.base import ContextOverflowError, LLMProvider, LLMUnavailableError, LLMUnreachableError
from app.repositories.documents import AppendixItemRow, DocumentRepository, RuleUnitRow
from app.schemas.ask import AskOutcome, AskResponse, Citation
from app.schemas.retrieval import Passage, RetrievalConfig, RetrievalResult, RetrievedEntity
from app.schemas.unit import UnitType
from app.services import verdict as _verdict
from app.services.grounding import (
    GroundingTrace,
    SupportCheck,
    filter_supported_citations,
    in_force_claims_without_in_force_document,
    legal_numbers,
    ungrounded_numbers,
)

logger = logging.getLogger(__name__)

GENERATION_RETRIEVAL_CONFIG = RetrievalConfig(k=5, passage_limit=15, entity_limit=12)
"""The retrieval config generation runs with -- ``passage_limit``/``entity_limit``
deliberately wider than ``RetrievalConfig``'s own default; ``k=5`` stated explicitly
(issue #169), chosen by measurement over a 4-point sweep (``eval.compare
--generation-k``/``--generation-passage-limit``) rather than left to fall through to
``RetrievalConfig``'s own bare default of the same value. See the module docstring's
own "k -- /ask's own seed width" section for the sweep table and why no wider point was
chosen instead."""

DISCLAIMER_VI = (
    "Thông tin trên chỉ mang tính chất tham khảo, được tổng hợp tự động từ các văn bản pháp "
    "luật hiện có trong hệ thống tại thời điểm tra cứu và có thể chưa đầy đủ hoặc chưa được "
    "cập nhật mới nhất. Vui lòng đối chiếu với văn bản gốc và tham vấn cơ quan, tổ chức có "
    "thẩm quyền trước khi áp dụng vào thực tế."
)
"""Fixed disclaimer text, attached to every ``AskOutcome.ANSWER`` response. C8 allows
Vietnamese here: this is user-facing product copy, not tooling."""

_SYSTEM_PROMPT_VI = (
    "Bạn là trợ lý tra cứu quy định pháp luật Việt Nam về kinh doanh khí dầu mỏ hóa lỏng "
    "(LPG/gas) và phòng cháy, chữa cháy, cứu nạn, cứu hộ (PCCC).\n\n"
    "QUY TẮC BẮT BUỘC, không có ngoại lệ:\n"
    "1. CHỈ được trả lời dựa trên NGỮ CẢNH PHÁP LÝ được cung cấp bên dưới. Không được dùng "
    "kiến thức nào khác ngoài NGỮ CẢNH, không được suy đoán, không được ước lượng con số.\n"
    "2. Mọi khẳng định trong câu trả lời PHẢI đi kèm ít nhất một trích dẫn (citations), và mỗi "
    'trích dẫn PHẢI dùng đúng một unit_id lấy CHÍNH XÁC từ các dòng "[unit_id: ...]" trong '
    "NGỮ CẢNH. TUYỆT ĐỐI không được bịa ra unit_id, số Điều, số Khoản nào không xuất hiện "
    "trong NGỮ CẢNH.\n"
    "3. Nếu NGỮ CẢNH không đủ thông tin để trả lời câu hỏi một cách chắc chắn, PHẢI từ chối "
    "(refusal=true, answer=null, citations=[]) thay vì suy đoán hoặc trả lời một phần không "
    "chắc chắn.\n"
    "4. NGỮ CẢNH có thể liệt kê các YÊU CẦU (obligations) đã được xác định sẵn, mỗi yêu cầu "
    "ghi rõ (các) unit_id NGUỒN của nó. Được phép dùng giá trị của yêu cầu đó trong câu trả "
    "lời, nhưng khi trích dẫn PHẢI trích dẫn (các) unit_id NGUỒN, không được trích dẫn chính "
    'mã yêu cầu (ví dụ không trích dẫn "req:...").\n'
    '5. KHÔNG được lặp lại nhãn "[unit_id: ...]" hay bất kỳ mã unit_id nào bên trong nội dung '
    'của "answer". Mã unit_id chỉ được đặt trong "citations" -- văn bản của "answer" chỉ chứa '
    "câu trả lời bằng tiếng Việt cho người đọc, không chứa mã định danh nội bộ nào.\n"
    '6. Nếu một đoạn NGỮ CẢNH có thêm các dòng "[unit_id: ...]" con cụ thể hơn (Khoản, Điểm) '
    "bên trong nó, PHẢI trích dẫn ĐÚNG MỘT unit_id con khớp chính xác với nội dung đang nêu, "
    "thay vì unit_id của cả đoạn. KHÔNG trích dẫn dồn nhiều unit_id con vào một trích dẫn, và "
    "KHÔNG trích dẫn mọi unit_id con được liệt kê nếu câu trả lời không thực sự dựa trên toàn "
    "bộ chúng -- chỉ trích dẫn (các) unit_id con thực sự chứa nội dung được nêu.\n\n"
    "CHỈ trả lời bằng một khối JSON đúng định dạng sau, không thêm bất kỳ văn bản, giải thích "
    "hay ký tự nào khác ngoài JSON:\n"
    '{"refusal": false, "answer": "<câu trả lời bằng tiếng Việt>", '
    '"citations": [{"unit_id": "<một unit_id có trong NGỮ CẢNH>"}]}\n'
    "hoặc, khi từ chối:\n"
    '{"refusal": true, "answer": null, "citations": []}'
)

_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "refusal": {"type": "boolean"},
        "answer": {"type": ["string", "null"]},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"unit_id": {"type": "string"}},
                "required": ["unit_id"],
            },
        },
    },
    "required": ["refusal", "answer", "citations"],
}
"""Passed to ``LLMProvider.generate`` as ``json_schema``: what a real provider that
supports structured decoding is asked to constrain its output to. Mirrors
``_ModelAnswer`` below field for field -- parsing still goes through that model
regardless, since a schema hint at decode time is not a guarantee (issue #90)."""


class _ModelCitation(BaseModel):
    """One citation as the model gave it -- not yet checked against anything."""

    model_config = ConfigDict(extra="ignore")

    unit_id: str = Field(min_length=1)


class _ModelAnswer(BaseModel):
    """The model's reply, parsed through a schema (C3) -- never scraped free text."""

    model_config = ConfigDict(extra="ignore")

    refusal: bool
    answer: str | None = None
    citations: list[_ModelCitation] = Field(default_factory=list)


_YES_NO_QUESTION_PATTERN = re.compile(
    r"\b(?:có|còn)\b.*\b(?:không|ko)\b(?:\s+(?:ạ|vậy|nhỉ))?\s*\??\s*$",
    re.IGNORECASE | re.DOTALL,
)
"""Matches a Vietnamese yes/no question (issue #173, widened by issue #198): a
"có"/"còn" particle anywhere in the sentence, ending in "không" or "ko", optionally
followed by one politeness/tag particle ("ạ"/"vậy"/"nhỉ"), optionally followed by a "?".
Covers every ending a shop owner actually types: "... không?", "... không", "... không
ạ?", "... không ạ", "... không vậy?", "... không vậy", "... không nhỉ?", "... không
nhỉ", "... ko?" and "... ko" -- in addition to the forms the pre-#198 pattern already
covered ("có phải ... không?", "có bắt buộc ... không?", "có cần ... không?", "có được
... không?", "có thuộc ... không?", "còn hiệu lực ... không?"). Diacritic-free input
("co phai ... khong") is out of scope (also touches retrieval; needs its own issue).

The closing anchor -- "không"/"ko", then at most one particle, then at most one "?",
then end of string -- is what keeps this from matching a mid-sentence "không," that is
not the question's own closing particle: ``refusal-tt-41-2011-status``'s "... còn áp
dụng không, quy định cụ thể ra sao?" has more text after its own mid-sentence "không,"
and is correctly not matched, since nothing in that sentence ends at "không"/"ko" itself
(``tests/test_generator.py``'s gold-detection test checks this against every row of
``data/gold.yaml``, not only this one row).

This exact form was tried once before (PR #175, commit ``74c1137``), measured unsafe on
the real ``/ask`` server and withdrawn, then re-applied here (issue #198) once
``app.services.verdict`` (issue #195, this module's own "Refusing when the answer's own
verdict ..." docstring section) gave a code-side backstop against the wrong verdict the
withdrawal was protecting against. See the PR body for both measurements in full."""


def _is_yes_no_question(question: str) -> bool:
    """Return whether ``question`` is answerable with Có/Không (issue #173).

    A pure function over the question text alone -- no model call, no retrieval, no
    corpus lookup. Used only to decide whether ``_build_user_prompt`` adds
    ``_YES_NO_INSTRUCTION_VI``; a non-match leaves the prompt exactly as it was before
    this issue existed.
    """
    return _YES_NO_QUESTION_PATTERN.search(question.strip()) is not None


_YES_NO_INSTRUCTION_VI = (
    "HƯỚNG DẪN RIÊNG CHO CÂU HỎI CÓ/KHÔNG:\n"
    'Câu hỏi dưới đây là câu hỏi dạng có/không. Câu trả lời ("answer") PHẢI:\n'
    '- bắt đầu bằng "Có", "Không" hoặc "Chỉ khi ..." (nêu rõ điều kiện ngay sau đó);\n'
    "- nêu rõ, bằng lời văn của câu trả lời, điều kiện/ngưỡng/số liệu mà (các) unit_id "
    "được trích dẫn quy định -- không chỉ trích dẫn suông mà không nói điều kiện đó là gì;\n"
    "- nếu câu hỏi đưa ra một số liệu cụ thể, câu trả lời PHẢI viết ra rõ phép so sánh bằng số "
    'giữa số liệu đó và ngưỡng nêu trong NGỮ CẢNH (ví dụ: "150 kg thấp hơn ngưỡng 200 kg"), rồi '
    "kết luận ĐÚNG theo phép so sánh vừa viết ra. Phép so sánh PHẢI dùng ĐÚNG quan hệ mà chính "
    "từ ngữ của đoạn NGỮ CẢNH nêu ra, KHÔNG được tự suy ra một quan hệ khác với từ ngữ đó:\n"
    '  * NGỮ CẢNH dùng "trên", "vượt quá" hoặc "lớn hơn" một mức: chỉ số liệu LỚN HƠN mức đó mới '
    'thỏa; một số liệu BẰNG mức đó KHÔNG thỏa điều kiện "trên" này, dù đoạn NGỮ CẢNH có nói đến '
    "nghĩa vụ đó ở mức cao hơn;\n"
    '  * NGỮ CẢNH dùng "từ ... trở lên", "không dưới" hoặc "tối thiểu" một mức: số liệu BẰNG '
    "hoặc LỚN HƠN mức đó mới thỏa;\n"
    '  * NGỮ CẢNH dùng "dưới" một mức: chỉ số liệu NHỎ HƠN mức đó mới thỏa; một số liệu BẰNG '
    'mức đó KHÔNG thỏa điều kiện "dưới" này;\n'
    '  * NGỮ CẢNH dùng "không quá" hoặc "tối đa" một mức: số liệu BẰNG hoặc NHỎ HƠN mức đó mới '
    "thỏa;\n"
    "- khi số liệu trong câu hỏi KHÔNG thỏa điều kiện theo đúng quan hệ trên, kết luận PHẢI là "
    '"Không" (không bắt buộc/không thuộc diện), TUYỆT ĐỐI không được kết luận "Có" trong trường '
    "hợp này dù đoạn NGỮ CẢNH có nói đến nghĩa vụ đó ở một mức khác;\n"
    "- KHÔNG được lặp lại câu hỏi dưới dạng một câu khẳng định để làm câu trả lời."
)
"""The instruction block ``_build_user_prompt`` adds right before "CÂU HỎI:", only when
``_is_yes_no_question`` matches (issue #173; the comparison rule below rewritten in its
fix round). Product content (C8 allows Vietnamese here): the model's own text must open
with Có/Không/Chỉ khi, state the cited clause's condition in its own words, and never
echo the question as the answer -- the exact failure #173 exists to fix (see the module
docstring's own "Answering a yes/no question" section).

The numeric-comparison rule is general, not tied to any one number: it tells the model to
apply the cited clause's own comparison word ("trên"/"vượt quá"/"lớn hơn" for strictly
greater, "từ ... trở lên"/"không dưới"/"tối thiểu" for at-or-above, "dưới"/"không
quá"/"tối đa" for the mirror lower-bound cases) literally, rather than the fix round's
own first wording, which hardcoded "BẰNG hoặc CAO HƠN ngưỡng" as always meaning "Có" --
wrong for Phụ lục VII mục 15's "... tồn chứa TRÊN 500 kg" (strictly greater), where a
number *equal* to the threshold does not meet it. The reviewer's real ``/ask`` run on
that first wording answered the new 500 kg boundary row "Có, ... phải mua ..." 3/3, a
confident, wrong "yes" -- see this file's own PR #175 body for the measured before/after
and the rejected first wording in full."""

_ARTICLE_NO = re.compile(r"#dieu-(\d+)")
_CLAUSE_NO = re.compile(r"#khoan-(\d+)")
_POINT_LETTER = re.compile(r"#diem-([^#]+)$")
"""Matches a point unit's own trailing ``#diem-<letter>`` segment, e.g. the ``a`` in
``...#dieu-8#khoan-1#diem-a``. A point is always a leaf (A5.1's own id scheme), so the
segment is always at the end of the id -- never a prefix check, which would also match
an appendix point's own ``#diem-...`` tail (``_label`` only ever applies this inside its
dieu/khoan branch, never its appendix one; see issue #187)."""
_APPENDIX_NO = re.compile(r"#phu-luc-(\d+)")
_APPENDIX_ITEM_NO = re.compile(r"#muc-(\d+)")
_APPENDIX_SUBITEM_NO = re.compile(r"#muc-\d+#muc-(\d+)$")
"""Matches a sub-item id's own trailing ``#muc-<n>`` segment (the part after the parent
item's own ``#muc-<n>``), e.g. the ``1`` in ``...#phu-luc-6#muc-35#muc-1``. Needed because
a sub-item's own number is not a column on ``units`` at all -- ``AppendixItemRow.item_no``
holds the *parent* item's number for this unit type -- so it is read from the id itself,
the same regex-on-the-locked-id-scheme approach ``app.schemas.ask.Citation`` already uses
for ``appendix_no``/``item_no``."""


@dataclass(frozen=True)
class _Source:
    """One unit the model was offered and may cite, with what a ``Citation`` needs.

    Built once per request from the retrieved context, independent of whether the
    unit reached the prompt as a passage's own chunk, a focus unit inside one, or a
    requirement's legal-basis source -- citation validation looks every model
    citation up here, and only here.

    ``body``, ``parent_id`` and ``ordinal`` (issue #187) are set only for a focus
    unit -- the data ``_build_user_prompt`` needs to render it as its own
    ``[unit_id: ...] <label>: <body>`` line, in the unit tree's own order, without a
    sixth parameter on that function (C10's five-argument ceiling). They play no part
    in a ``Citation`` and default to empty/``None``/``0`` for every other source.
    """

    unit_id: str
    document_id: str
    document_number: str
    doc_type: str
    dieu_no: int | None
    khoan_no: int | None
    effective: date | None
    possibly_superseded: bool
    body: str = ""
    parent_id: str | None = None
    ordinal: int = 0


def _parse_unit_id(unit_id: str) -> tuple[str, int | None, int | None]:
    """Return ``unit_id``'s owning document id, article number and clause number.

    Parsed from the id itself (A5.1's own scheme) rather than trusting a
    ``Passage``'s ``dieu_no``/``khoan_no``, which describe the *chunk* -- often a
    whole article -- while a focus unit inside it can name one specific clause.
    """
    document_id = unit_id.split("#", 1)[0]
    article = _ARTICLE_NO.search(unit_id)
    clause = _CLAUSE_NO.search(unit_id)
    if _APPENDIX_NO.search(unit_id):
        return document_id, None, None
    return (
        document_id,
        int(article.group(1)) if article else None,
        int(clause.group(1)) if clause else None,
    )


async def _build_sources(
    passages: Sequence[Passage],
    entities: Sequence[RetrievedEntity],
    documents: DocumentRepository,
) -> dict[str, _Source]:
    """Return every unit the prompt offers as citable, keyed by unit id.

    A passage's own unit id and every one of its focus units share that passage's
    document and ``possibly_superseded`` flag (both are document-level facts); an
    entity's source that is not already covered that way still becomes citable, with
    ``possibly_superseded`` defaulting to ``False`` -- a flag is only ever raised on
    evidence, never assumed.

    Also resolves every passage's own focus units' ``body``/``parent_id``/``ordinal``
    (``DocumentRepository.get_focus_unit_contexts``, issue #187), onto the same
    ``_Source`` entry -- so ``_build_user_prompt`` can render each as its own line
    without a database call, and without a sixth parameter on that function. A
    passage's own unit id is added to that same query (fix round on #187): a
    passage's own ``body`` is non-empty for every clause-level chunk and for an
    article chunk with an intro, and ``_render_passage_lines`` needs it to render that
    text once, alongside its children -- fetched here, in the one batched query, never
    a second round trip per passage.
    """
    flagged: dict[str, bool] = {}
    for passage in passages:
        flagged[passage.unit_id] = passage.possibly_superseded
        for focus in passage.focus_units:
            flagged.setdefault(focus, passage.possibly_superseded)
    for entity in entities:
        for source in entity.sources:
            flagged.setdefault(source, False)

    document_ids = {unit_id.split("#", 1)[0] for unit_id in flagged}
    rows = await documents.get_many(sorted(document_ids))

    focus_ids = {focus for passage in passages for focus in passage.focus_units}
    focus_ids.update(passage.unit_id for passage in passages)
    focus_contexts = await documents.get_focus_unit_contexts(sorted(focus_ids))

    sources: dict[str, _Source] = {}
    for unit_id, possibly_superseded in flagged.items():
        document_id, dieu_no, khoan_no = _parse_unit_id(unit_id)
        row = rows.get(document_id)
        if row is None:
            continue  # A dangling reference: no document row, nothing safe to cite.
        focus_context = focus_contexts.get(unit_id)
        sources[unit_id] = _Source(
            unit_id=unit_id,
            document_id=document_id,
            document_number=row.number,
            doc_type=row.doc_type,
            dieu_no=dieu_no,
            khoan_no=khoan_no,
            effective=row.effective,
            possibly_superseded=possibly_superseded,
            body=focus_context.body if focus_context else "",
            parent_id=focus_context.parent_id if focus_context else None,
            ordinal=focus_context.ordinal if focus_context else 0,
        )
    return sources


_PREMIUM_TABLE_APPENDIX_NO = 6
"""Phụ lục VI's own appendix number. Duplicated from ``ingest.parser.APPENDIX_6_NO``
rather than imported -- ``app/`` never depends on ``ingest/`` (``ingest`` imports
from ``app``, not the reverse), so this module keeps its own copy, the same
deliberate duplication ``ingest/chunk.py``'s own ``_PREMIUM_TABLE_APPENDIX_NO`` alias
already models one layer down."""

_DEDUCTIBLE_CELL_PATTERN = re.compile(r"\b([MN])\s+\d+,\d+")
"""Matches a Phụ lục VI premium-table row's own deductible-type cell, e.g. "N 0,3" or
"M 0,05" (issue #165) -- read straight off a passage's own already-retrieved text,
never typed into this module. Specific enough that it does not fire on a lookalike
substring in a different appendix's item (e.g. Phụ lục VII mục 15's "... trên 500 kg
N 0,3", which this module never even reaches: ``_with_deductible_rule_passages``
gates on ``_PREMIUM_TABLE_APPENDIX_NO`` first)."""


def _deductible_types(text: str) -> list[str]:
    """Return the distinct deductible-type letters (``M``/``N``) found in ``text``.

    Sorted so the order is deterministic regardless of which cell the source happens
    to print first; a row can rarely carry both letters with no sub-row split at all.
    """
    return sorted({match.group(1) for match in _DEDUCTIBLE_CELL_PATTERN.finditer(text)})


_DEDUCTIBLE_RULE_PERCENT = re.compile(r"tối đa là ([\d,]+)\s*%")
"""Reads the deductible cap's own percentage straight off a rule unit's stored text
(e.g. "... mức khấu trừ bảo hiểm tối đa là 10% số tiền bảo hiểm ..."), for
``_deductible_rule_label`` -- never a number typed into this module."""


def _deductible_rule_label(letter: str, rule_unit_body: str) -> str:
    """Build the display label for a resolved deductible-type rule unit (issue #165).

    Replaces the generic, uninformative label ``_label`` would otherwise give an
    ``appendix_section_point`` unit (no mục number to show, only the document number)
    with one that actually says which letter and cap it is -- read from the rule
    unit's own stored text, never hardcoded. A row's "M"/"N" cell carries no number a
    model can already see means "the rate", so without this, nothing in the prompt
    tells it that *this* unit, among the Mục II khoản 1's several điểm, is the one
    naming *that* letter's own percentage cap.

    Args:
        letter: The deductible type this rule unit states ("M" or "N").
        rule_unit_body: The rule unit's own stored text.

    Returns:
        E.g. ``'Phụ lục VI, Mục II Khoản 1 -- ký hiệu N, mức khấu trừ tối đa 10%'``,
        or the same without the percentage clause when none could be read off the
        text (a defensive fallback, never a crash).
    """
    label = f"Phụ lục VI, Mục II Khoản 1 -- ký hiệu {letter}"
    percent_match = _DEDUCTIBLE_RULE_PERCENT.search(rule_unit_body)
    if percent_match is not None:
        label += f", mức khấu trừ tối đa {percent_match.group(1)}%"
    return label


async def _with_deductible_rule_passages(
    passages: Sequence[Passage], documents: DocumentRepository
) -> tuple[tuple[Passage, ...], dict[str, str]]:
    """Offer each premium-table row's own deductible-type rule as its own citable unit.

    Issue #165 (fix round on #166): a Phụ lục VI row's "M"/"N" cell names a deductible
    type, but the row's own text never states that type's actual rule -- the real cap
    (1% for M, 10% for N) lives on a separate điểm of the appendix's own Mục II khoản
    1. A citation is only ever valid when the cited unit's own text supports the claim
    (``_run_grounding``, completely unchanged by this function), so the model needs
    that điểm offered as its own citable unit, not just the row that names the letter.

    Two earlier versions of this fix were tried and measured against the real model
    (qwen3:8b) before this one, both regressing a previously-correct answer:

    1. Appending the rule's text into the row's own *chunk* at embedding time diluted
       every row's embedding and regressed the flagship and premium questions -- see
       the module docstring. Reverted; ``ingest/chunk.py`` is byte-identical with
       `main`.
    2. Appending a one-line pointer ("Ký hiệu N: mức khấu trừ theo [unit_id: ...]")
       onto the row's own *passage text* at generation time fixed the deductible
       question but regressed the premium question every time it was tried --
       measured across several phrasings, including a minimal, content-free filler
       line with no mention of "mức khấu trừ" at all, which did NOT regress it. The
       trigger is not length but content: any text that recasts what the row's
       passage is "about" toward the deductible rule makes the model stop treating
       that same row as the answer to a premium-rate question, even though it still
       is one. Confirmed not to be `main`-side flicker: `main`, re-run fresh in the
       same session, kept citing the premium row correctly every time.
    3. A third version kept text untouched but added a brand-new, standalone
       ``Passage`` for the resolved rule unit *in addition to* linking it as a
       ``focus_units`` entry on the triggering row. That combination intermittently
       regressed the **flagship** question (5/8, later 6/8 of 8 fresh runs, vs 8/8 for
       `main` and for either half of the combination tried alone) whenever the
       flagship's own retrieval happened to also pull in a Phụ lục VI row (muc-17/18
       rank just below its real Phụ lục VII answer) -- an emergent interaction, not
       reproducible from either half in isolation. Dropped; see point 4.

    This version only ever does two things, confirmed stable across repeated runs
    (8-12 each) of the flagship, premium and both deductible questions, with zero
    regressions: it adds the resolved rule unit's id to the triggering row's own
    ``focus_units`` (issue #108's own mechanism for a finer-grained citable sub-unit,
    never the row's ``text``), and it gives that unit a clear, value-bearing display
    label via ``deductible_rule_labels`` (``_build_user_prompt``'s own parameter) --
    never a brand-new ``Passage``. The label turned out to be the missing piece a
    bare focus-unit link alone did not supply: offering the unit with no label (or
    the generic one ``_label`` gives an ``appendix_section_point``, which carries no
    mục number) still let the model cite it while stating the *other* letter's
    percentage, which the grounding verifier correctly caught and refused -- safe,
    but not a fix. ``_deductible_rule_label`` reads the actual percentage off the
    rule unit's own stored text, so the label itself never states a number this
    module invented.

    Scoped to Phụ lục VI alone: a passage from any other appendix (e.g. Phụ lục VII,
    whose own items can coincidentally carry an "N 0,3"-shaped substring) is never
    touched, since its unit id never names ``_PREMIUM_TABLE_APPENDIX_NO``.

    Args:
        passages: The passages ``generate_answer`` is about to offer the model,
            already capped to ``GENERATION_RETRIEVAL_CONFIG.passage_limit``.
        documents: Looks up each deductible type's rule unit
            (``DocumentRepository.get_deductible_rule_units``).

    Returns:
        A pair of:

        * ``passages`` unchanged when no row carries an M/N cell. Otherwise, each
          matching row gets the resolved rule unit id added to its own
          ``focus_units`` (never its ``text``) -- so the model is offered a direct,
          finer-grained citation path from that row to the rule that governs its
          cell. Never adds, removes or reorders a ``Passage``.
        * A ``{unit_id: label}`` map (``_build_user_prompt``'s own
          ``deductible_rule_labels`` argument) naming the display label for every
          resolved rule unit, built by ``_deductible_rule_label`` -- read off the
          rule unit's own text, never a bare, uninformative generic label.
    """
    rule_units_by_document: dict[str, dict[str, RuleUnitRow]] = {}
    labels: dict[str, str] = {}

    async def resolve(document_id: str, letter: str) -> RuleUnitRow | None:
        rule_units = rule_units_by_document.get(document_id)
        if rule_units is None:
            rule_units = await documents.get_deductible_rule_units(
                document_id, _PREMIUM_TABLE_APPENDIX_NO
            )
            rule_units_by_document[document_id] = rule_units
        return rule_units.get(letter)

    updated: list[Passage] = []
    for passage in passages:
        appendix_match = _APPENDIX_NO.search(passage.unit_id)
        is_premium_table_row = (
            appendix_match is not None
            and int(appendix_match.group(1)) == _PREMIUM_TABLE_APPENDIX_NO
        )
        letters = _deductible_types(passage.text) if is_premium_table_row else []
        if not letters:
            updated.append(passage)
            continue

        new_focus_units = [
            rule_unit_id
            for letter in letters
            if (
                rule_unit_id := await _resolve_deductible_rule_unit_id(
                    passage, letter, resolve, labels
                )
            )
            is not None
            and rule_unit_id not in passage.focus_units
        ]
        if new_focus_units:
            passage = passage.model_copy(
                update={"focus_units": (*passage.focus_units, *new_focus_units)}
            )
        updated.append(passage)

    return tuple(updated), labels


async def _resolve_deductible_rule_unit_id(
    row: Passage,
    letter: str,
    resolve: Callable[[str, str], Awaitable[RuleUnitRow | None]],
    labels: dict[str, str],
) -> str | None:
    """Resolve one (row, letter) pair for ``_with_deductible_rule_passages``.

    Factored out so that function's own branching stays under the project's
    complexity ceiling (``ruff``'s C901). Updates ``labels`` in place (keyed by the
    rule unit's id, via ``_deductible_rule_label``) as a side effect.

    Returns:
        The resolved rule unit's id, so the caller can add it to ``row``'s own
        ``focus_units``; ``None`` when no rule unit was found for ``letter`` (already
        logged here).
    """
    rule_unit = await resolve(row.document_id, letter)
    if rule_unit is None:
        logger.warning(
            "generator: %s names deductible type %s but no rule unit for it was found "
            "in %s's Phụ lục VI; its cell is offered with no rule to cite (issue #165)",
            row.unit_id,
            letter,
            row.document_id,
        )
        return None
    labels[rule_unit.id] = _deductible_rule_label(letter, rule_unit.body)
    return rule_unit.id


_ROMAN_NUMERAL_VALUES: tuple[tuple[int, str], ...] = (
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
)


def _roman_numeral(value: int) -> str:
    """Return ``value`` in uppercase Roman numerals (an appendix number never exceeds a few tens).

    Shared by ``_label`` (the citation label) and ``_appendix_quote_label`` (issue #146's
    verbatim-quote label) so both spell the same appendix the same way, from the same one
    conversion.
    """
    remainder = value
    numeral = ""
    for amount, symbol in _ROMAN_NUMERAL_VALUES:
        count, remainder = divmod(remainder, amount)
        numeral += symbol * count
    return numeral


_DOC_TYPE_LABEL_VI: dict[str, str] = {
    "law": "Luật",
    "decree": "Nghị định",
    "circular": "Thông tư",
    "tcvn": "TCVN",
    "qcvn": "QCVN",
    "unknown": "",
}
"""Vietnamese display word for each ``DocType`` value (``DocumentRow.doc_type``, a raw
stored string -- see its own docstring). Keyed by the enum's plain string value rather
than importing ``DocType`` itself, matching how ``DocumentRow`` already carries it."""


def _label(source: _Source) -> str:
    """Return a short citation label for one source, e.g. "Văn bản ... , Điều 53, Khoản 1".

    A point unit (``...#diem-x``) gets its own ", Điểm x" after the Khoản, so two
    points of the same clause no longer carry identical labels (issue #187) -- only
    inside the dieu/khoan branch, never the appendix one: an appendix point's own
    label (``deductible_rule_labels`` territory, issue #165) is out of scope here and
    stays exactly as before.

    Used only in the prompt (``_build_user_prompt``), never on a user-facing
    ``Citation`` -- ``app.schemas.ask.Citation`` builds its own display fields
    straight from ``dieu_no``/``khoan_no``/``appendix_no``/``item_no``, independent
    of this function.
    """
    parts = [f"Văn bản {source.document_number}"]
    appendix = _APPENDIX_NO.search(source.unit_id)
    item = _APPENDIX_ITEM_NO.search(source.unit_id)
    if appendix is not None:
        label = f"Phụ lục {_roman_numeral(int(appendix.group(1)))}"
        if item is not None:
            label += f", mục {int(item.group(1))}"
        label += f" — {source.document_number}"
        parts = [label]
    elif source.dieu_no is not None:
        parts.append(f"Điều {source.dieu_no}")
    if source.khoan_no is not None:
        parts.append(f"Khoản {source.khoan_no}")
    point = _POINT_LETTER.search(source.unit_id)
    if appendix is None and point is not None:
        parts.append(f"Điểm {point.group(1)}")
    if source.effective is not None:
        parts.append(f"hiệu lực từ {source.effective.isoformat()}")
    return ", ".join(parts)


def _ordered_focus_units(
    root_id: str, focus_units: Sequence[str], sources: dict[str, _Source]
) -> list[str]:
    """Order ``focus_units`` the way the unit tree nests them under ``root_id``.

    Parent before child, siblings by ``ordinal`` (issue #187's own trap): never by
    sorting the id strings themselves, which does not match numeric order
    (``khoan-10`` sorts before ``khoan-2`` as text). Walks down from ``root_id``
    using each unit's own ``_Source.parent_id`` (``_build_sources``'s own
    ``get_focus_unit_contexts`` lookup), so a multi-level passage (clause, then its
    points) comes out in reading order even though ``focus_units`` itself is not
    guaranteed to be.

    A unit whose tree position could not be resolved (no ``_Source`` entry, or no
    ``parent_id`` on it -- a dangling reference, never expected in practice since a
    focus unit is itself a real, retrieved unit) is appended at the end, in its
    original order, rather than dropped: every focus unit still gets its own citable
    line.
    """
    children_of: dict[str, list[tuple[int, str]]] = {}
    for unit_id in focus_units:
        source = sources.get(unit_id)
        if source is None or source.parent_id is None:
            continue
        children_of.setdefault(source.parent_id, []).append((source.ordinal, unit_id))

    def walk(parent_id: str) -> list[str]:
        ordered: list[str] = []
        for _, unit_id in sorted(children_of.get(parent_id, [])):
            ordered.append(unit_id)
            ordered.extend(walk(unit_id))
        return ordered

    ordered = walk(root_id)
    seen = set(ordered)
    ordered.extend(unit_id for unit_id in focus_units if unit_id not in seen)
    return ordered


def _passage_label(
    unit_id: str, sources: dict[str, _Source], deductible_rule_labels: dict[str, str]
) -> str:
    """Return the display label for one citable unit -- a passage header or a focus line alike."""
    source = sources.get(unit_id)
    return deductible_rule_labels.get(unit_id) or (_label(source) if source else unit_id)


def _render_passage_lines(
    passage: Passage, sources: dict[str, _Source], deductible_rule_labels: dict[str, str]
) -> list[str]:
    """Return one passage's own prompt lines, not including the blank line after it.

    An appendix passage (``#phu-luc``), or one with no focus units, renders exactly
    as it always has: its own ``[unit_id: ...]`` label line, the flattened ``text``
    verbatim, then (when it has focus units) an indented block naming each one by its
    own label alone -- no body (issue #108's own shape; issue #187 never touches
    this).

    Any other passage that has focus units -- an article or clause whose "1." / "a)"
    markers ``ingest`` strips out of ``text`` -- instead drops that flattened text
    and renders its heading (``text``'s first line), then its own ``body`` exactly
    once when it is non-empty (a fix round on #187: an article's own ``body`` is
    empty only for *some* articles -- every clause-level chunk, and an article chunk
    with an intro, has a non-empty one, and the first round's render dropped it),
    then one line per focus unit, in the unit tree's own order
    (``_ordered_focus_units``), each line showing that unit's own body exactly once:
    ``[unit_id: <id>] <label>: <body>``. Never both: rendering a unit's body *and*
    keeping it inside the flattened text duplicated it, which overflowed the model's
    context on a real gold row (issue #187's own probe, variant B).
    """
    label = _passage_label(passage.unit_id, sources, deductible_rule_labels)
    lines = [f"[unit_id: {passage.unit_id}] {label}"]
    is_appendix = _APPENDIX_NO.search(passage.unit_id) is not None
    if is_appendix or not passage.focus_units:
        lines.append(passage.text)
        if passage.focus_units:
            lines.append("Các đơn vị cụ thể hơn bên trong đoạn trên:")
            for focus_unit in passage.focus_units:
                focus_label = _passage_label(focus_unit, sources, deductible_rule_labels)
                lines.append(f"  [unit_id: {focus_unit}] {focus_label}")
        return lines

    stripped_text = passage.text.strip()
    if stripped_text:
        lines.append(stripped_text.splitlines()[0])
    own_source = sources.get(passage.unit_id)
    own_body = " ".join((own_source.body if own_source else "").split())
    if own_body:
        lines.append(own_body)
    for focus_unit in _ordered_focus_units(passage.unit_id, passage.focus_units, sources):
        focus_label = _passage_label(focus_unit, sources, deductible_rule_labels)
        focus_source = sources.get(focus_unit)
        body = " ".join((focus_source.body if focus_source else "").split())
        lines.append(f"[unit_id: {focus_unit}] {focus_label}: {body}")
    return lines


def _build_user_prompt(
    question: str,
    passages: Sequence[Passage],
    entities: Sequence[RetrievedEntity],
    sources: dict[str, _Source],
    deductible_rule_labels: dict[str, str] | None = None,
) -> str:
    """Build the user prompt: the retrieved context, then the question.

    Every citable unit -- a passage's own id and every one of its ``focus_units`` --
    is introduced by its own ``[unit_id: ...]`` line, the exact marker ``MockLLM``
    reads back too (see its module docstring) and the only shape
    ``_SYSTEM_PROMPT_VI``'s rule 2 tells the model a citation may come from. A focus
    unit's line carries its own label (Điều/Khoản/Điểm, from ``sources``) rather than
    the bare id alone, so a model choosing between several siblings of the same
    passage has more to go on than an unlabelled list (issue #108); for an article or
    clause passage (never an appendix one), that line also carries the unit's own
    body text, once, in place of the passage's flattened text repeating it under a
    bare label (issue #187 -- see ``_render_passage_lines``).

    When ``question`` is a yes/no question (``_is_yes_no_question``), ``_YES_NO_
    INSTRUCTION_VI`` is inserted right before the "CÂU HỎI:" line (issue #173); for
    every other question, nothing here differs from before that issue existed.

    Args:
        question: The question, exactly as asked.
        passages: Passages to render, in order.
        entities: Requirement/concept entities to render below the passages.
        sources: Every citable unit's ``_Source``, for its default label.
        deductible_rule_labels: Issue #165's own override, ``{unit_id: label}`` for a
            Phụ lục VI deductible-type rule unit (``_with_deductible_rule_passages``,
            ``_deductible_rule_label``) -- replaces ``sources``' own generic label
            (an ``appendix_section_point`` has no mục number to show) with one that
            actually names the letter and percentage cap, wherever that unit's line
            is rendered: as its own passage, or as a focus unit under the premium-table
            row whose cell names it.
    """
    deductible_rule_labels = deductible_rule_labels or {}
    lines = ["NGỮ CẢNH PHÁP LÝ:", ""]
    for passage in passages:
        lines.extend(_render_passage_lines(passage, sources, deductible_rule_labels))
        lines.append("")
    if entities:
        lines.append("CÁC YÊU CẦU (obligations) LIÊN QUAN ĐÃ ĐƯỢC XÁC ĐỊNH SẴN:")
        for entity in entities:
            value = entity.meta.get("value", entity.meta) if entity.meta else "(không có giá trị)"
            source_ids = ", ".join(entity.sources) if entity.sources else "không có"
            lines.append(f"- {entity.label_vi}: {value} (unit_id nguồn: {source_ids})")
        lines.append("")
    if _is_yes_no_question(question):
        lines.append(_YES_NO_INSTRUCTION_VI)
        lines.append("")
    lines.append(f"CÂU HỎI: {question}")
    return "\n".join(lines)


def _parse_model_reply(raw: str) -> _ModelAnswer | None:
    """Parse the model's raw reply through ``_ModelAnswer``; ``None`` if it does not fit."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("generator: model reply was not valid JSON: %r", raw[:200])
        return None
    try:
        return _ModelAnswer.model_validate(payload)
    except ValidationError as error:
        logger.warning("generator: model reply did not match the expected schema: %s", error)
        return None


def _split_bundled_citations(
    model_citations: Sequence[_ModelCitation],
) -> list[_ModelCitation]:
    """Split a citation whose ``unit_id`` bundles several ids into one comma-joined string.

    A local qwen2.5 quirk noted since #107's own PR and confirmed across the full gold set
    while diagnosing #108: the model sometimes returns several real, individually-citable
    unit ids packed into a single JSON ``unit_id`` field --
    e.g. ``"gas-decree-87-2018#dieu-53#khoan-3, gas-decree-87-2018#dieu-53#khoan-2"`` -- instead
    of one ``{"unit_id": ...}`` object per id. Never a real unit id itself: A5.1's own scheme
    (``_UNIT_ID_SEGMENT``) never puts a comma inside one id, so any comma is proof of a bundle,
    never a false split of something that was already a single valid id.

    This runs *before* ``_validate_citations``, never instead of it: every piece produced here
    is still looked up against ``sources`` exactly as any other citation would be, so a bundle
    made of nonsense ids is still fully rejected -- this only stops a bundle of otherwise-valid
    ids from being rejected as one unrecognisable string, it does not skip or weaken the
    fabrication check itself in any way.

    Args:
        model_citations: The model's own citations, exactly as parsed from its reply.

    Returns:
        One ``_ModelCitation`` per non-blank, comma-separated segment of each input
        citation's ``unit_id``. A citation with no comma yields exactly one, unchanged,
        result -- so a normal reply is not altered in count or order.
    """
    split: list[_ModelCitation] = []
    for citation in model_citations:
        for piece in citation.unit_id.split(","):
            piece = piece.strip()
            if piece:
                split.append(_ModelCitation(unit_id=piece))
    return split


def _validate_citations(
    model_citations: Sequence[_ModelCitation], sources: dict[str, _Source]
) -> tuple[tuple[Citation, ...], bool]:
    """Check every model citation against ``sources``; drop what was not actually offered.

    Returns:
        The citations that survived, in the model's own order with duplicates
        collapsed, and whether any of them carries ``possibly_superseded``.
    """
    kept: dict[str, Citation] = {}
    possibly_superseded = False
    for item in model_citations:
        source = sources.get(item.unit_id)
        if source is None:
            logger.warning(
                "generator: dropped a fabricated citation -- %r was not in the retrieved "
                "context for this question",
                item.unit_id,
            )
            continue
        kept[source.unit_id] = Citation(
            unit_id=source.unit_id,
            document_id=source.document_id,
            document_number=source.document_number,
            dieu_no=source.dieu_no,
            khoan_no=source.khoan_no,
            effective=source.effective,
        )
        possibly_superseded = possibly_superseded or source.possibly_superseded
    return tuple(kept.values()), possibly_superseded


_UNIT_ID_SEGMENT = r"[a-z0-9]+(?:-[a-z0-9]+)*"
"""One id segment in A5.1's scheme, e.g. ``dieu-53`` or ``gas-decree-87-2018``."""

_UNIT_ID_SHAPE = (
    rf"{_UNIT_ID_SEGMENT}(?:[{re.escape(UNIT_SEPARATOR)}{re.escape(ENTITY_SEPARATOR)}]"
    rf"{_UNIT_ID_SEGMENT})*"
)
"""A full id in A5.1's scheme: one segment, then zero or more ``#``/``:``-separated
segments -- e.g. ``gas-decree-87-2018#dieu-53#khoan-1`` or ``req:min-store-area``.
Built from ``app.core.ids``'s own separators rather than hard-coded punctuation, so
this stays in step with the id scheme it is meant to recognise."""

_LEAKED_UNIT_ID_MARKER = re.compile(
    rf"\s?[(\[{{]?\s*unit_id\s*:\s*{_UNIT_ID_SHAPE}\s*[)\]}}]?",
    re.IGNORECASE,
)
"""Matches a ``[unit_id: ...]`` marker (or a plausible variant: different bracket
style, no bracket at all, extra or no surrounding space) wherever it appears in a
string. The only fixed anchor is the literal token ``unit_id:``, which never occurs
in ordinary Vietnamese prose, so this cannot match legitimate parenthesised text such
as "(trừ trường hợp có tường ngăn cháy)" -- that text does not contain the token, so
the pattern never starts matching there at all. The bracket characters on each side
are independently optional so a marker missing its closing bracket (e.g. a truncated
model reply) is still recognised and removed."""


def _strip_leaked_unit_id_markers(answer: str) -> str:
    """Remove any ``[unit_id: ...]``-shaped marker that leaked into generated prose.

    A deterministic backstop for ``_SYSTEM_PROMPT_VI``'s rule 5: the instruction not
    to repeat a unit_id in ``answer`` is a request, not a guarantee (the same
    reasoning issue #90 applied to citations), so this always runs over the model's
    ``answer`` text regardless of whether it followed that instruction. It only ever
    touches ``answer`` -- ``citations`` is built independently, from ``sources``, and
    is never passed through this function.

    Args:
        answer: The model's own answer text, before it is returned to the caller.

    Returns:
        ``answer`` with every leaked marker removed, extra runs of spaces left
        behind by a removal collapsed to one, and the ends trimmed.
    """
    stripped = _LEAKED_UNIT_ID_MARKER.sub("", answer)
    stripped = re.sub(r"[ \t]{2,}", " ", stripped)
    return stripped.strip()


_MAX_QUOTABLE_BODY_LENGTH = 300
"""Issue #146's own "short means" rule: an appendix item/sub-item longer than this is
never quoted (and never truncated to fit -- a shortened legal quote would say something
the source does not), only logged."""


def _appendix_quote_label(
    source: _Source, appendix_no: int, item_no: int, subitem_no: int | None
) -> str:
    """Return the label a verbatim appendix quote is introduced by (issue #146).

    e.g. ``"Phụ lục VII, mục 15 (Nghị định 105/2025/NĐ-CP)"``. The Roman numeral and the
    document number both come from data -- ``appendix_no``/``item_no``/``subitem_no`` from
    the unit's own stored coordinates (``AppendixItemRow``, never the model's text) and
    ``source.document_number``/``source.doc_type`` from the owning document's own row
    (``_build_sources``) -- never from anything the model said.

    Args:
        source: The citation's source, for its document number and type.
        appendix_no: The appendix's 1-based position (``7`` labels as ``VII``).
        item_no: The item's number inside the appendix.
        subitem_no: The sub-item's number inside the item, or ``None`` for a plain item.

    Returns:
        The label, with no trailing colon or quote marks.
    """
    doc_type_word = _DOC_TYPE_LABEL_VI.get(source.doc_type, "")
    document_label = f"{doc_type_word} {source.document_number}".strip()
    muc = f"mục {item_no}"
    if subitem_no is not None:
        muc += f", mục {subitem_no}"
    return f"Phụ lục {_roman_numeral(appendix_no)}, {muc} ({document_label})"


def _quote_cited_appendix_items(
    answer: str,
    citations: Sequence[Citation],
    appendix_rows: dict[str, AppendixItemRow],
    sources: dict[str, _Source],
) -> str:
    """Append one verbatim quote per qualifying cited appendix item/sub-item (issue #146).

    Deterministic post-processing, never an LLM call: runs only on ``citations`` that
    already survived fabrication validation and the grounding check (including #158's
    status guard), and only ever reads ``AppendixItemRow.body`` and each source's own
    document data -- nothing here can add a citation, drop one, or change the model's
    own answer text beyond appending lines to it.

    Only an ``"appendix_item"``/``"appendix_subitem"`` row of at most
    ``_MAX_QUOTABLE_BODY_LENGTH`` characters is quoted. ``appendix_rows`` already excludes
    every other unit type -- a form, an article, a clause, a Roman prose section, or a
    lettered point on its own -- by construction (``DocumentRepository.get_appendix_items``'s
    own SQL filter); the ``row.unit_type`` check below is a second, defense-in-depth layer
    that never trusts a caller to have filtered correctly, the same reasoning issue #90
    applied to citation fabrication. A longer item is logged and skipped, never truncated:
    a shortened legal quote would say something the source does not.

    Args:
        answer: The answer text, already stripped of leaked unit_id markers.
        citations: The final citations on the response, in citation order.
        appendix_rows: Appendix item/sub-item rows among the cited ids, keyed by unit id.
        sources: The per-request source map (``_build_sources``), for each citation's
            document number and type.

    Returns:
        ``answer`` unchanged -- byte for byte -- when no citation qualifies; otherwise
        ``answer`` with one quote line appended per qualifying citation, in citation
        order, each cited unit quoted at most once.
    """
    lines: list[str] = []
    quoted: set[str] = set()
    for citation in citations:
        if citation.unit_id in quoted:
            continue
        row = appendix_rows.get(citation.unit_id)
        source = sources.get(citation.unit_id)
        if row is None or source is None or row.appendix_no is None or row.item_no is None:
            continue
        if row.unit_type not in (UnitType.APPENDIX_ITEM.value, UnitType.APPENDIX_SUBITEM.value):
            continue
        subitem_no: int | None = None
        if row.unit_type == UnitType.APPENDIX_SUBITEM.value:
            subitem_match = _APPENDIX_SUBITEM_NO.search(row.id)
            if subitem_match is None:
                continue  # malformed id -- nothing safe to label a sub-item with
            subitem_no = int(subitem_match.group(1))
        if len(row.body) > _MAX_QUOTABLE_BODY_LENGTH:
            logger.info(
                "generator: not quoting %s -- body is %d characters, over the %d-character limit",
                citation.unit_id,
                len(row.body),
                _MAX_QUOTABLE_BODY_LENGTH,
            )
            continue
        quoted.add(citation.unit_id)
        label = _appendix_quote_label(source, row.appendix_no, row.item_no, subitem_no)
        lines.append(f'{label}: "{row.body}"')
    if not lines:
        return answer
    return answer + "\n" + "\n".join(lines)


async def _quote_appendix_citations_if_any(
    answer: str | None,
    citations: Sequence[Citation],
    sources: dict[str, _Source],
    documents: DocumentRepository,
    grounding_trace: GroundingTrace | None = None,
) -> str | None:
    """Apply ``_quote_cited_appendix_items`` to ``answer``, or return it unchanged.

    Factored out of ``generate_answer`` so that function stays under the project's own
    complexity ceiling (``ruff``'s C901), the same reasoning that already factored out
    ``_run_grounding``/``_generate_or_refuse``. ``answer`` is ``None`` only when grounding
    is disabled and the model's own reply carried no answer text -- nothing to quote.

    Args:
        answer: The answer text, already stripped of leaked unit_id markers, after
            grounding (when enabled) has settled ``citations``.
        citations: The final citations on the response, in citation order.
        sources: The per-request source map (``_build_sources``).
        documents: Looks up the appendix item/sub-item rows among ``citations``.
        grounding_trace: Optional side-channel for evaluation (issue #167); records
            ``answer`` as ``pre_quote_answer`` before anything below appends to it, the
            same reason this function runs where it does in ``generate_answer`` -- after
            fabrication validation and the grounding check, before the corpus's own quoted
            text can be mistaken for the model's claim.

    Returns:
        ``answer`` unchanged when it is ``None`` or no citation qualifies for a quote;
        otherwise ``answer`` with each qualifying citation's quote line appended.
    """
    if grounding_trace is not None:
        grounding_trace.pre_quote_answer = answer
    if answer is None:
        return None
    appendix_rows = await documents.get_appendix_items(tuple(c.unit_id for c in citations))
    return _quote_cited_appendix_items(answer, citations, appendix_rows, sources)


def _refuse(reason: str) -> AskResponse:
    """Build the refusal response for one reason code (see ``AskResponse.refusal_reason``)."""
    return AskResponse(outcome=AskOutcome.REFUSAL, refusal_reason=reason)


def _format_number(value: float) -> str:
    """Render a parsed quantity/threshold for the verdict-check log line (issue #195).

    Every number ``app.services.verdict`` parses is a plain count ("500 kg", "12 m2"), so
    this drops a whole number's own trailing ``.0`` (``500.0`` -> ``"500"``); a genuinely
    fractional value (e.g. ``2.5`` tấn) prints as Python's own ``str`` would.
    """
    return str(int(value)) if value == int(value) else str(value)


def _log_verdict_contradiction(facts: _verdict.VerdictFacts) -> None:
    """Log one ``verdict_contradicts_clause`` refusal -- parsed facts only (issue #195, trap 4).

    Never the question or answer text: the holdout runner must never be able to echo a row's
    own text back out of a log line, the same reason every other refusal log in this module
    names ids and numbers, never prose.
    """
    logger.warning(
        "generator: verdict contradicts clause -- quantity %s %s, threshold %s %s %s, "
        "expected %s, answer opened %s",
        _format_number(facts.quantity),
        facts.unit,
        facts.op,
        _format_number(facts.threshold),
        facts.unit,
        facts.expected,
        facts.opened,
    )


def _refuse_on_verdict_contradiction(
    question: str,
    citations: Sequence[Citation],
    answer: str | None,
    sources: dict[str, _Source],
) -> AskResponse | None:
    """Refuse with ``"verdict_contradicts_clause"``, or ``None`` to let the caller continue.

    Issue #195: the answer's own opening verdict contradicted its own cited clause.
    Factored out of ``generate_answer`` so that function stays under this repository's own
    complexity ceiling (``ruff``'s C901) -- the same reasoning that already factored out
    ``_run_grounding``/``_generate_or_refuse``.

    Checks every kept citation's OWN unit text (``_Source.body``, the same per-unit line issue
    #188 shows in the prompt), never an article's recomposed text, which usually states several
    thresholds and would make the check silently never fire. Runs unconditionally (not gated on
    ``_is_yes_no_question``): ``app.services.verdict.evaluate`` itself only ever fires on an
    answer that opens with a verdict and a question that states exactly one quantity, so it is
    already a no-op on every other path -- see the module docstring's own "Refusing when the
    answer's own verdict contradicts its own cited clause" section.

    Args:
        question: The question, exactly as asked.
        citations: The citations that survived fabrication validation and, when enabled, the
            grounding check -- what the final response would cite.
        answer: The model's own answer text, or ``None`` when grounding is disabled and the
            model gave no answer text at all -- nothing to check in that case.
        sources: The per-request source map (``_build_sources``), for each citation's own text.
    """
    if answer is None:
        return None
    clause_texts = [sources[c.unit_id].body for c in citations if c.unit_id in sources]
    verdict_result, verdict_facts = _verdict.evaluate(question, clause_texts, answer)
    if verdict_result != "contradiction":
        return None
    assert verdict_facts is not None  # evaluate's own contract: set iff "contradiction"
    _log_verdict_contradiction(verdict_facts)
    return _refuse("verdict_contradicts_clause")


@dataclass(frozen=True)
class _FinalizationRequest:
    """What ``_finalize_citations`` needs about one answer, bundled into one argument.

    Exists so ``_finalize_citations`` takes this, ``documents``, ``llm`` and
    ``grounding_trace`` as its own separate arguments rather than seven positional ones -- the
    same PLR0913 reasoning ``_GroundingRequest``/``SupportCheck`` already apply.
    """

    question: str
    answer: str | None
    citations: tuple[Citation, ...]
    passages: tuple[Passage, ...]
    sources: dict[str, _Source]


async def _finalize_citations(
    request: _FinalizationRequest,
    documents: DocumentRepository,
    llm: LLMProvider,
    grounding_trace: GroundingTrace | None,
) -> tuple[Citation, ...] | AskResponse:
    """Run the grounding check (when enabled) and then the verdict check (issue #195).

    Factored out of ``generate_answer`` so that function stays under this repository's own
    complexity ceiling (``ruff``'s C901) -- the same reasoning that already factored out
    ``_run_grounding``/``_generate_or_refuse``. Bundles both checks into one call because both
    only ever act on the same, final ``citations`` -- in order: the grounding filter runs
    first (so the verdict check reads whatever fewer, more-trustworthy citations survive it),
    then the verdict check runs against the citations the grounding step actually kept (or,
    when ``GROUNDING_CHECK=false``, the ones fabrication validation already kept).

    Returns:
        The final ``citations`` tuple when nothing refuses; otherwise the ``AskResponse``
        refusal -- ``"unsupported_citations"`` (grounding is on but there is no answer text to
        check), whatever ``_run_grounding`` itself returned, or ``"verdict_contradicts_clause"``.
    """
    citations = request.citations
    if get_settings().grounding_check:
        if request.answer is None:
            return _refuse("unsupported_citations")
        grounded = await _run_grounding(
            _GroundingRequest(request.question, request.answer, citations, request.passages),
            documents,
            llm,
            grounding_trace,
        )
        if isinstance(grounded, AskResponse):
            return grounded
        citations = grounded

    verdict_refusal = _refuse_on_verdict_contradiction(
        request.question, citations, request.answer, request.sources
    )
    if verdict_refusal is not None:
        return verdict_refusal

    return citations


async def _generate_or_refuse(
    llm: LLMProvider,
    *,
    system: str,
    user: str,
    json_schema: dict[str, Any],
    grounding_trace: GroundingTrace | None,
) -> tuple[str, None] | tuple[None, AskResponse]:
    """Call ``llm.generate`` once, converting a provider failure into a refusal response.

    Factored out of ``generate_answer`` so that function stays under the project's own
    complexity ceiling (``ruff``'s C901) without suppressing the check -- the same reasoning
    that already factored out ``_grounded_citations``/``_run_grounding``.

    Returns:
        ``(raw_reply, None)`` on success, or ``(None, refusal_response)`` when the call
        raised ``LLMUnavailableError`` or ``ContextOverflowError`` (issue #129).
    """
    try:
        raw = await llm.generate(system=system, user=user, json_schema=json_schema)
    except (LLMUnavailableError, ContextOverflowError) as error:
        return None, _refuse_for_llm_error(error)
    if grounding_trace is not None:
        # Read right away, before any later verifier call through the same provider can
        # overwrite it -- see ``OllamaProvider.last_prompt_eval_count``'s own docstring.
        # ``MockLLM`` carries no such attribute, so this is ``None`` for it, exactly as
        # intended (issue #129's own acceptance criterion: "MockLLM is unaffected").
        grounding_trace.prompt_eval_count = getattr(llm, "last_prompt_eval_count", None)
    return raw, None


async def _run_grounding(
    request: "_GroundingRequest",
    documents: DocumentRepository,
    llm: LLMProvider,
    trace: "GroundingTrace | None",
) -> "tuple[Citation, ...] | AskResponse":
    """Run ``_grounded_citations`` and convert both its outcomes into one return type.

    Factored out of ``generate_answer`` for the same complexity reason as
    ``_generate_or_refuse`` above: this absorbs the try/except for a provider failure during
    grounding (issue #129) and the "nothing survived -> refuse" branch into one call, so the
    caller only ever branches on "is this a refusal already" once.
    """
    try:
        grounded = await _grounded_citations(request, documents, llm, trace)
    except (LLMUnavailableError, ContextOverflowError) as error:
        return _refuse_for_llm_error(error)
    if isinstance(grounded, str):
        return _refuse(grounded)
    return grounded if grounded is not None else _refuse("unsupported_citations")


def _refuse_for_llm_error(error: LLMUnavailableError | ContextOverflowError) -> AskResponse:
    """Turn a provider-level call failure into its own refusal, never an unhandled 500.

    Issues #129/#183: every exception here means "this call cannot be trusted or reached",
    never that the corpus or the model's own judgement found nothing -- each gets a distinct
    reason code so a frontend (and this project's own evaluation) can tell a context-window
    problem, a connection failure and a transient timeout apart.

    ``LLMUnreachableError`` is checked before the plain ``LLMUnavailableError`` branch since it
    is a subclass of it (issue #183's own trap) -- a timeout alone keeps "llm_unavailable" (#112
    already relies on that code meaning "the model itself was slow"), while a connection failure
    gets its own "llm_unreachable" rather than being folded into that measurement.
    """
    if isinstance(error, ContextOverflowError):
        logger.error("generator: refused -- %s", error)
        return _refuse("context_overflow")
    if isinstance(error, LLMUnreachableError):
        logger.error("generator: refused -- the LLM provider was unreachable: %s", error)
        return _refuse("llm_unreachable")
    logger.error("generator: refused -- the LLM provider was unavailable: %s", error)
    return _refuse("llm_unavailable")


@dataclass(frozen=True)
class _GroundingRequest:
    """What ``_grounded_citations`` needs about one answer, bundled into one argument.

    Exists so ``_grounded_citations`` takes this, ``documents``, ``llm`` and ``trace`` as its
    own separate arguments rather than seven positional parameters (the PR review: this is
    what used to carry a ``# noqa: PLR0913``).
    """

    question: str
    answer: str
    citations: tuple[Citation, ...]
    passages: tuple[Passage, ...]


async def _grounded_citations(
    request: _GroundingRequest,
    documents: DocumentRepository,
    llm: LLMProvider,
    trace: GroundingTrace | None,
) -> tuple[Citation, ...] | str | None:
    """Run both grounding layers and return the citations that survive, or ``None`` to refuse.

    Factored out of ``generate_answer`` so that function stays under the project's own
    complexity ceiling without suppressing the check (the PR review: this used to carry a
    ``# noqa: C901``). Bundles layer 1 (the deterministic document-number guard) and layer 2
    (the per-citation verifier) together since both need the same ``source_texts``.

    Args:
        request: The question, the model's answer (already stripped of leaked
            ``[unit_id: ...]`` markers), the citations that already passed fabrication
            validation, and the passages ``generate_answer`` offered the model (so a cited
            unit's text falls back to its passage when ``documents`` has nothing for it).
        documents: Looks up each cited id's full text (its own, plus every descendant's).
        llm: The provider the per-citation verifier asks.
        trace: Optional side-channel for evaluation; records the layer-2 decisions, each
            verifier call's latency, and whether layer 1 refused.

    Returns:
        The citations layer 2 kept, in their original order; ``"status_unknown"``
        when the status guard refuses; or ``None`` when the existing document-number
        guard refuses or layer 2 keeps nothing.
    """
    source_texts = {passage.unit_id: passage.text for passage in request.passages}
    for passage in request.passages:
        for focus_unit in passage.focus_units:
            source_texts.setdefault(focus_unit, passage.text)
    source_texts.update(
        await documents.get_unit_contexts(tuple(c.unit_id for c in request.citations))
    )

    status_rows = await documents.get_by_numbers(tuple(legal_numbers(request.answer)))
    unverified_status_claims = in_force_claims_without_in_force_document(
        request.answer, {row.number: row.status for row in status_rows}
    )
    if unverified_status_claims:
        logger.warning(
            "generator: refused -- answer claims unverified document(s) are in force: %s",
            sorted(unverified_status_claims),
        )
        if trace is not None:
            trace.layer1_refusals += 1
        return "status_unknown"

    ungrounded = ungrounded_numbers(request.answer, request.citations, source_texts)
    if ungrounded:
        logger.warning(
            "generator: refused -- answer names ungrounded document number(s): %s",
            sorted(ungrounded),
        )
        if trace is not None:
            trace.layer1_refusals += 1
        return None

    check = SupportCheck(
        question=request.question, answer=request.answer, source_texts=source_texts
    )
    supported = await filter_supported_citations(check, request.citations, llm, trace)
    return supported or None


@dataclass(frozen=True)
class _GenerationContext:
    """What every attempt inside ``generate_answer`` needs (issue #202).

    Independent of which passages that attempt itself offers. Bundled into one object so
    ``_generate_attempt``/``_retry_without_rejected_passages`` take
    this, their own ``passages``/``attempt`` and ``trace`` as separate parameters rather than
    up to seven positional ones -- the same PLR0913 reasoning ``_FinalizationRequest``/
    ``_GroundingRequest``/``SupportCheck`` already apply.
    """

    question: str
    entities: tuple[RetrievedEntity, ...]
    documents: DocumentRepository
    llm: LLMProvider
    deductible_rule_labels: dict[str, str]


@dataclass(frozen=True)
class _Attempt:
    """The result of one generation attempt inside ``generate_answer`` (issue #202).

    The first call, or the one retry this issue allows -- either way, bundles what both the
    retry decision and the final response need, so ``generate_answer`` never duplicates either
    attempt's own logic.

    Attributes:
        passages: The passages THIS attempt offered the model -- the full, generation-capped
            list on the first attempt, the reduced one on a retry. Used by ``_retry_without_
            rejected_passages`` to tell a rejected *passage* id (removable) from a rejected
            *focus-unit* id (not, per the issue's own trap).
        sources: This attempt's own ``_build_sources`` map, for the final response's quoting
            step -- rebuilt on a retry since fewer passages make fewer units citable.
        refusal: The ``AskResponse`` this attempt refused with, or ``None`` when it answered.
        answer: The model's own answer text (after marker stripping), set whenever a reply was
            parsed far enough to have one -- including a grounding/verdict refusal, so ``_retry_
            without_rejected_passages`` never needs it, but kept for symmetry with the success
            case. ``None`` on a refusal from before parsing (e.g. a provider failure).
        citations: The final, grounding- and verdict-checked citations, only set when ``refusal``
            is ``None``.
        possibly_superseded: Only meaningful when ``refusal`` is ``None``.
        rejected_unit_ids: Unit ids the per-citation verifier rejected during this attempt's own
            finalization -- empty unless ``refusal.refusal_reason == "unsupported_citations"``
            AND layer 2 actually ran and kept nothing (a layer-1 refusal or a missing answer
            never call the verifier at all, so this stays empty for them by construction, with
            no separate layer check needed).
    """

    passages: tuple[Passage, ...]
    sources: dict[str, _Source]
    refusal: AskResponse | None
    answer: str | None = None
    citations: tuple[Citation, ...] = ()
    possibly_superseded: bool = False
    rejected_unit_ids: tuple[str, ...] = ()


async def _generate_attempt(
    context: _GenerationContext, passages: Sequence[Passage], trace: GroundingTrace
) -> _Attempt:
    """Run one full generation attempt: build the prompt, call, parse, validate, finalize.

    Factored out of ``generate_answer`` so issue #202's one retry can run this exact same
    sequence a second time, over a reduced ``passages``, with nothing skipped and nothing
    duplicated -- the retry goes through every step the first attempt does, including the
    fabrication check (so a reduced context is never silently exempt from it) and the
    grounding/verdict checks.
    """
    passages = tuple(passages)
    sources = await _build_sources(passages, context.entities, context.documents)
    user_prompt = _build_user_prompt(
        context.question, passages, context.entities, sources, context.deductible_rule_labels
    )
    raw, refusal = await _generate_or_refuse(
        context.llm,
        system=_SYSTEM_PROMPT_VI,
        user=user_prompt,
        json_schema=_JSON_SCHEMA,
        grounding_trace=trace,
    )
    if refusal is not None:
        return _Attempt(passages=passages, sources=sources, refusal=refusal)
    assert raw is not None  # narrowed by _generate_or_refuse's own (raw, None) | (None, refusal)

    parsed = _parse_model_reply(raw)
    if parsed is None:
        return _Attempt(passages=passages, sources=sources, refusal=_refuse("invalid_model_output"))
    if parsed.refusal:
        return _Attempt(passages=passages, sources=sources, refusal=_refuse("model_declined"))
    if not parsed.citations:
        return _Attempt(passages=passages, sources=sources, refusal=_refuse("no_citations_given"))

    citations, possibly_superseded = _validate_citations(
        _split_bundled_citations(parsed.citations), sources
    )
    if not citations:
        return _Attempt(passages=passages, sources=sources, refusal=_refuse("no_valid_citations"))

    answer = _strip_leaked_unit_id_markers(parsed.answer) if parsed.answer else parsed.answer
    decisions_before = len(trace.decisions)
    finalized = await _finalize_citations(
        _FinalizationRequest(context.question, answer, citations, passages, sources),
        context.documents,
        context.llm,
        trace,
    )
    if isinstance(finalized, AskResponse):
        rejected = tuple(
            unit_id for unit_id, supported in trace.decisions[decisions_before:] if not supported
        )
        return _Attempt(
            passages=passages,
            sources=sources,
            refusal=finalized,
            answer=answer,
            citations=citations,
            rejected_unit_ids=rejected,
        )

    return _Attempt(
        passages=passages,
        sources=sources,
        refusal=None,
        answer=answer,
        citations=finalized,
        possibly_superseded=possibly_superseded,
    )


async def _retry_without_rejected_passages(
    context: _GenerationContext, trace: GroundingTrace, attempt: _Attempt
) -> _Attempt:
    """Retry ``attempt`` once without its rejected passages, or return it unchanged (issue #202).

    Retries only when every one of these holds -- each a deliberate exclusion, not an
    oversight (see the module docstring's own "Retrying once ..." section and the issue's own
    traps):

    * ``attempt`` refused with ``"unsupported_citations"``. A layer-1 refusal (the document-
      number guard, or the status guard's ``"status_unknown"``), ``"verdict_contradicts_
      clause"``, ``"model_declined"``, ``"no_citations_given"``, ``"no_valid_citations"``,
      ``"invalid_model_output"``, an LLM-error reason, or ``"no_context"`` never retries.
    * ``attempt.rejected_unit_ids`` is non-empty -- which, by ``_generate_attempt``'s own
      construction, is only ever true when layer 2 (``filter_supported_citations``) actually
      ran and rejected everything it was given. A layer-1 refusal or a missing answer never
      reaches the verifier at all, so this is empty for them without a separate layer check.
    * At least one rejected id is the own ``unit_id`` of a passage ``attempt`` actually
      offered -- never a citation that was only a focus unit *inside* a passage (the passage
      itself was never rejected, so there is nothing of its own to remove).
    * Removing those passages would still leave something to answer from (a passage or an
      entity) -- never a retry over empty context, which would be indistinguishable from the
      first call ``generate_answer`` itself already refuses as ``"no_context"``.

    The retry removes only the rejected passages and nothing else: ``context.entities`` is
    unchanged, and no new or different passage is ever added back in their place.
    """
    if attempt.refusal is None or attempt.refusal.refusal_reason != "unsupported_citations":
        return attempt
    if not attempt.rejected_unit_ids:
        return attempt

    passage_ids = {passage.unit_id for passage in attempt.passages}
    drop_ids = {unit_id for unit_id in attempt.rejected_unit_ids if unit_id in passage_ids}
    if not drop_ids:
        return attempt

    reduced_passages = tuple(p for p in attempt.passages if p.unit_id not in drop_ids)
    if not reduced_passages and not context.entities:
        return attempt

    logger.info("generator: retrying once without rejected passage(s): %s", sorted(drop_ids))
    trace.retried = True
    return await _generate_attempt(context, reduced_passages, trace)


async def generate_answer(
    question: str,
    retrieval: RetrievalResult,
    llm: LLMProvider,
    documents: DocumentRepository,
    grounding_trace: GroundingTrace | None = None,
) -> AskResponse:
    """Generate a cited answer for ``question`` from ``retrieval``'s context, or refuse.

    Never called on the emergency path (``app.services.safety`` intercepts before
    this). Calls ``llm`` at most twice for generation (issue #202): a second time only
    when the per-citation verifier rejects every citation of the first attempt and at
    least one rejected id is a passage that was actually offered -- see
    ``_retry_without_rejected_passages``'s own docstring for the exact condition, and the
    module docstring's "Retrying once ..." section for why. Never called at all when
    ``retrieval`` found nothing to answer from -- an empty corpus match is refused
    immediately, since there is nothing a model could ground an answer in.

    Issue #129: a provider timeout (``LLMUnavailableError``) or a reply whose
    ``prompt_eval_count`` shows it was truncated or near enough to it (``ContextOverflowError``)
    is never allowed to propagate as an unhandled exception (which ``app.api.ask`` would
    otherwise surface as an HTTP 500) -- both are caught here, around the main generation call
    and around the grounding verifier's calls alike, and turned into their own refusal reasons.

    Args:
        question: The question, exactly as asked.
        retrieval: What ``Retriever.retrieve`` found for it.
        llm: The provider to ask -- ``MockLLM`` or ``OllamaProvider``.
        documents: Used to look up each cited unit's document number and effective
            date, since neither a ``Passage`` nor a ``RetrievedEntity`` carries a
            citable unit's own document number.
        grounding_trace: Optional side-channel for evaluation; it never changes the public
            ``AskResponse``. Issue #202: when the caller leaves this ``None`` (e.g. ``/ask``),
            an internal trace is still used to drive the retry decision -- it is simply
            discarded afterward, since nothing reads it back.

    Returns:
        An ``AskOutcome.ANSWER`` response with at least one verified citation, or
        an ``AskOutcome.REFUSAL`` response naming why.
    """
    passages = retrieval.passages[: GENERATION_RETRIEVAL_CONFIG.passage_limit]
    entities = retrieval.entities[: GENERATION_RETRIEVAL_CONFIG.entity_limit]
    if not passages and not entities:
        return _refuse("no_context")

    trace = grounding_trace if grounding_trace is not None else GroundingTrace()

    # Issue #165: offer each Phụ lục VI row's own deductible-type rule as its own
    # citable passage, never changing retrieval or a chunk -- see the module docstring.
    passages, deductible_rule_labels = await _with_deductible_rule_passages(passages, documents)

    context = _GenerationContext(question, entities, documents, llm, deductible_rule_labels)
    attempt = await _generate_attempt(context, passages, trace)
    attempt = await _retry_without_rejected_passages(context, trace, attempt)

    if attempt.refusal is not None:
        return attempt.refusal

    # Issue #146: deterministic, post-grounding quote of any cited appendix item/sub-item
    # short enough to quote verbatim -- runs on the final ``citations`` (after both the
    # fabrication check and, when enabled, the grounding check including #158's status
    # guard), never before, and never calls the model. Issue #167: records the model's own
    # pre-quote answer onto ``grounding_trace``, for eval.compare's deciding-fact check.
    answer = await _quote_appendix_citations_if_any(
        attempt.answer, attempt.citations, attempt.sources, documents, trace
    )

    return AskResponse(
        outcome=AskOutcome.ANSWER,
        answer=answer,
        citations=attempt.citations,
        disclaimer=DISCLAIMER_VI,
        possibly_superseded=attempt.possibly_superseded,
    )
