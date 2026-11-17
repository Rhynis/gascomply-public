"""Typed cross-references detected by regex (build plan A6 Phase 1.4).

A ``Reference`` is one draft row for ``references.csv``: a candidate edge for
human and graph review in Phase 2.2. Nothing here is resolved to a real
document or unit identifier -- ``target_hint`` is a normalized string only.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ReferenceKind(StrEnum):
    """Category of an in-text cross-reference, driving its Phase 2.2 edge type.

    Attributes:
        BASIS: A "Căn cứ ..." citation, becomes BASED_ON / GUIDED_BY.
        AMENDMENT: An amendment or repeal cue, becomes AMENDED_BY / REPLACED_BY.
        INTERNAL: A same-document article reference, becomes REFERENCES within
            one document.
        EXTERNAL: A document or standard number, becomes REFERENCES across
            documents.
    """

    BASIS = "basis"
    AMENDMENT = "amendment"
    INTERNAL = "internal"
    EXTERNAL = "external"


class Reference(BaseModel):
    """One draft cross-reference row, matched by regex.

    Attributes:
        src_id: Identifier of the unit that contains the reference, or the
            document identifier when the reference came from the preamble.
        kind: Category of the reference.
        raw_text: The matched span, verbatim from the source.
        target_hint: Normalized target, e.g. ``87/2018/NĐ-CP``,
            ``TCVN 6223:2017``, or ``Điều 38 Khoản 1``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    src_id: str = Field(min_length=1)
    kind: ReferenceKind
    raw_text: str = Field(min_length=1)
    target_hint: str = Field(min_length=1)
