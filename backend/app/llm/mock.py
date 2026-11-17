"""Offline stand-in for a real ``LLMProvider`` (build plan A6 Phase 3.6, issue #90).

Not a language model: it runs no inference and calls no network. Like
``MockEmbedder`` (issue #89's own precedent), it exists so the whole pipeline --
prompt construction, JSON parsing, citation validation, the API layer -- runs and is
tested with nothing external, and stays the ``LLM_PROVIDER`` default so ``python -m
app`` and CI both work with no Ollama anywhere.

It must still answer with *something a real provider could plausibly have said*, in
the exact JSON shape ``generator.py`` parses, or none of the pipeline downstream of
it is exercised. To do that without any actual understanding of the question, it
reads the context ``generator.py`` itself placed in the prompt -- specifically the
``[unit_id: ...]`` marker ``generator.py`` writes in front of every passage and
source it offers the model -- and cites the first few. This is not the free-text
citation-scraping the issue forbids: that rule is about trusting a *model's own
invented* output as if it were real; this reads a fixed marker this codebase's own
prompt-builder writes, the same way ``MockEmbedder`` derives its vector from a hash
of its own input rather than doing anything semantic.

Issue #101: the marker is never alone on its line. ``_build_user_prompt`` writes it
as ``f"[unit_id: {passage.unit_id}] {label}"`` -- the source's label always follows
on the *same* line, e.g. ``[unit_id: gas-decree-87-2018#dieu-53] Văn bản 87/2018/NĐ-CP,
Điều 53``. A regex anchored to end-of-line (``$``) never matches that, so this
provider matched nothing, cited nothing, and refused every question -- including with
no code change since, because nothing ever exercised it against a real, generator-built
prompt (``tests/test_llm_mock.py`` used hand-written one-marker-per-line strings that
happened to satisfy the old, wrong anchor). ``_UNIT_ID_MARKER`` below anchors only on
the literal ``[unit_id: `` prefix and the closing ``]``, matching wherever on the line
it appears; ``tests/test_llm_mock.py`` binds this to reality by building a prompt
through the real ``_build_user_prompt`` and asserting extraction from it, so a future
change to that line's shape breaks the test instead of silently reintroducing this bug.
"""

import json
import re
from typing import Any

from app.llm.base import LLMProvider

_UNIT_ID_MARKER = re.compile(r"\[unit_id: ([^\]\s]+)\]")
"""Matches this codebase's own ``[unit_id: <id>]`` marker anywhere it appears on a
line, not only when it is the whole line -- ``_build_user_prompt`` always follows it
with the source's label on the same line (issue #101). The id itself never contains
``]`` or whitespace (``app.core.ids``'s scheme: alphanumerics, ``-``, ``#``, ``:``),
so ``[^\\]\\s]+`` captures exactly the id and nothing past the closing bracket."""

_MAX_CITATIONS = 3
"""How many of the offered units this stand-in cites -- enough to exercise a
multi-citation answer without depending on how many the context happens to hold."""

_ANSWER_VI = (
    "Theo ngữ cảnh được cung cấp, nội dung liên quan đến câu hỏi nằm trong các "
    "điều khoản được trích dẫn dưới đây."
)


class MockLLM(LLMProvider):
    """Deterministic, offline double for ``LLMProvider`` -- see the module docstring."""

    async def generate(
        self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
    ) -> str:
        """Return a canned, schema-shaped reply, citing the first units offered.

        Args:
            system: Ignored -- there is no model to instruct.
            user: Scanned for this codebase's own ``[unit_id: ...]`` markers, unless
                ``json_schema`` is the grounding verifier's own schema.
            json_schema: The generator's reply is scanned for ``[unit_id: ...]`` markers;
                the grounding verifier's (identified by its own ``supported`` property,
                never by a marker written into the prompt -- issue #118's own follow-up
                review) is always answered ``{"supported": true}``.

        Returns:
            A JSON string: a refusal when ``user`` offers no unit at all (nothing
            to found an answer on), otherwise an answer citing up to
            ``_MAX_CITATIONS`` of the offered units, in the order they appear.
        """
        if json_schema is not None and "supported" in json_schema.get("properties", {}):
            return json.dumps({"supported": True})
        unit_ids = list(dict.fromkeys(_UNIT_ID_MARKER.findall(user)))[:_MAX_CITATIONS]
        if not unit_ids:
            reply: dict[str, Any] = {"refusal": True, "answer": None, "citations": []}
        else:
            reply = {
                "refusal": False,
                "answer": _ANSWER_VI,
                "citations": [{"unit_id": unit_id} for unit_id in unit_ids],
            }
        return json.dumps(reply, ensure_ascii=False)
