"""Tests pinning the stable identifier scheme (build plan A5.1).

Phase 2 and Phase 3 join on these strings, so any change to the format is a
breaking change and must fail here first.
"""

import pytest

from app.core.ids import (
    EntityType,
    appendix_form_id,
    appendix_id,
    appendix_item_id,
    appendix_point_id,
    appendix_section_clause_id,
    appendix_section_id,
    appendix_section_point_id,
    appendix_subitem_id,
    article_id,
    chapter_id,
    clause_id,
    document_id,
    entity_id,
    point_id,
)

DOC = "gas-decree-87-2018"


def test_identifier_format_is_locked() -> None:
    assert document_id(DOC) == "gas-decree-87-2018"
    assert chapter_id(DOC, 3) == "gas-decree-87-2018#chuong-3"
    assert article_id(DOC, 38) == "gas-decree-87-2018#dieu-38"
    assert clause_id(DOC, 38, 1) == "gas-decree-87-2018#dieu-38#khoan-1"
    assert point_id(DOC, 38, 1, "a") == "gas-decree-87-2018#dieu-38#khoan-1#diem-a"
    assert appendix_id(DOC, 1) == "gas-decree-87-2018#phu-luc-1"
    assert appendix_item_id(DOC, 1, 13) == "gas-decree-87-2018#phu-luc-1#muc-13"
    assert appendix_point_id(DOC, 1, 13, "a") == "gas-decree-87-2018#phu-luc-1#muc-13#diem-a"
    assert appendix_subitem_id(DOC, 1, 35, 1) == "gas-decree-87-2018#phu-luc-1#muc-35#muc-1"
    assert appendix_form_id(DOC, 1, "01") == "gas-decree-87-2018#phu-luc-1#mau-01"


def test_entity_namespaces_are_disjoint_from_unit_identifiers() -> None:
    assert entity_id(EntityType.REQUIREMENT, "min-store-area") == "req:min-store-area"
    assert entity_id(EntityType.CONCEPT, "lpg-retail-store") == "concept:lpg-retail-store"
    assert entity_id(EntityType.FORM, "license-application") == "form:license-application"
    assert entity_id(EntityType.AUTHORITY, "ubnd-huyen") == "authority:ubnd-huyen"
    assert entity_id(EntityType.FEE, "license-fee") == "fee:license-fee"

    unit_ids = {chapter_id(DOC, 1), article_id(DOC, 1), clause_id(DOC, 1, 1)}
    entity_ids = {entity_id(kind, "x") for kind in EntityType}
    assert not unit_ids & entity_ids
    assert all(":" not in unit for unit in unit_ids)
    assert all("#" not in entity for entity in entity_ids)


def test_identifiers_are_deterministic_and_unique_per_node() -> None:
    assert article_id(DOC, 38) == article_id(DOC, 38)

    generated = [
        chapter_id(DOC, 3),
        article_id(DOC, 38),
        clause_id(DOC, 38, 1),
        clause_id(DOC, 38, 2),
        point_id(DOC, 38, 1, "a"),
        point_id(DOC, 38, 1, "b"),
        article_id("pccc-law-2024", 38),
    ]
    assert len(set(generated)) == len(generated)


def test_clause_identifier_is_prefixed_by_its_article() -> None:
    assert clause_id(DOC, 38, 1).startswith(f"{article_id(DOC, 38)}#")
    assert point_id(DOC, 38, 1, "a").startswith(f"{clause_id(DOC, 38, 1)}#")


def test_vietnamese_point_letter_is_accepted_and_normalized() -> None:
    assert point_id(DOC, 5, 2, "đ").endswith("#diem-đ")
    assert point_id(DOC, 5, 2, " A ") == point_id(DOC, 5, 2, "a")


def test_form_code_is_lowercased_and_trimmed() -> None:
    assert appendix_form_id(DOC, 8, "PC01") == appendix_form_id(DOC, 8, "pc01")
    assert appendix_form_id(DOC, 8, " PC01 ") == appendix_form_id(DOC, 8, "pc01")
    assert appendix_form_id(DOC, 1, "01a") == "gas-decree-87-2018#phu-luc-1#mau-01a"


def test_form_identifier_is_a_sibling_of_an_appendix_item_not_a_child() -> None:
    assert appendix_form_id(DOC, 1, "01") == f"{appendix_id(DOC, 1)}#mau-01"
    assert not appendix_form_id(DOC, 1, "01").startswith(appendix_item_id(DOC, 1, 1))


def test_subitem_identifier_is_a_child_of_its_own_parent_item() -> None:
    assert appendix_subitem_id(DOC, 1, 35, 1) == f"{appendix_item_id(DOC, 1, 35)}#muc-1"
    assert appendix_subitem_id(DOC, 1, 35, 1) != appendix_subitem_id(DOC, 1, 35, 2)
    # A different parent item's own sub-item 1 is a distinct identifier --
    # the sub-item number is local to its parent, not a running count.
    assert appendix_subitem_id(DOC, 1, 35, 1) != appendix_subitem_id(DOC, 1, 36, 1)


@pytest.mark.parametrize("code", ["", "pc 01", "pc-01", "pc_01", "PC!01"])
def test_invalid_form_code_is_rejected(code: str) -> None:
    with pytest.raises(ValueError, match="Invalid form code"):
        appendix_form_id(DOC, 1, code)


@pytest.mark.parametrize(
    "slug",
    ["Gas-Decree", "gas decree", "gas--decree", "-gas", "gas#decree", "req:gas", ""],
)
def test_invalid_document_slug_is_rejected(slug: str) -> None:
    with pytest.raises(ValueError, match="Invalid document slug"):
        document_id(slug)


@pytest.mark.parametrize("number", [0, -1])
def test_non_positive_unit_numbers_are_rejected(number: int) -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        article_id(DOC, number)
    with pytest.raises(ValueError, match="must be >= 1"):
        chapter_id(DOC, number)
    with pytest.raises(ValueError, match="must be >= 1"):
        clause_id(DOC, 1, number)
    with pytest.raises(ValueError, match="must be >= 1"):
        appendix_subitem_id(DOC, 1, number, 1)
    with pytest.raises(ValueError, match="must be >= 1"):
        appendix_subitem_id(DOC, 1, 35, number)


@pytest.mark.parametrize("diem", ["1", "a)", "", "abc", "A1"])
def test_invalid_point_letter_is_rejected(diem: str) -> None:
    with pytest.raises(ValueError, match="Invalid point letter"):
        point_id(DOC, 1, 1, diem)


def test_invalid_entity_slug_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid entity slug"):
        entity_id(EntityType.REQUIREMENT, "Min Store Area")


# --- Phụ lục VI's own Roman-numbered section prose (issue #147) -----------------


def test_appendix_section_identifier_format_is_locked() -> None:
    assert appendix_section_id(DOC, 6, "I") == "gas-decree-87-2018#phu-luc-6#muc-i"
    assert appendix_section_id(DOC, 6, "II") == "gas-decree-87-2018#phu-luc-6#muc-ii"
    assert (
        appendix_section_clause_id(DOC, 6, "I", 1) == "gas-decree-87-2018#phu-luc-6#muc-i#khoan-1"
    )
    assert (
        appendix_section_point_id(DOC, 6, "II", 1, "a")
        == "gas-decree-87-2018#phu-luc-6#muc-ii#khoan-1#diem-a"
    )


def test_appendix_section_identifiers_never_collide_with_a_table_row() -> None:
    # Row 1 ("Trụ sở làm việc...") and Mục I's own "1." paragraph are a
    # genuine ambiguity in the source document (issue #147's own review) --
    # the section scheme's point is that they get different ids.
    assert appendix_section_id(DOC, 6, "I") != appendix_item_id(DOC, 6, 1)
    assert appendix_section_clause_id(DOC, 6, "I", 1) != appendix_item_id(DOC, 6, 1)


def test_appendix_section_clause_and_point_are_children_of_their_own_parent() -> None:
    assert appendix_section_clause_id(DOC, 6, "I", 1).startswith(
        f"{appendix_section_id(DOC, 6, 'I')}#"
    )
    assert appendix_section_point_id(DOC, 6, "II", 1, "a").startswith(
        f"{appendix_section_clause_id(DOC, 6, 'II', 1)}#"
    )
    # A clause's own id is local to its section: the same clause number
    # under a different section is a distinct identifier.
    assert appendix_section_clause_id(DOC, 6, "I", 1) != appendix_section_clause_id(DOC, 6, "II", 1)


def test_appendix_section_label_is_normalized_case_insensitively() -> None:
    assert appendix_section_id(DOC, 6, "i") == appendix_section_id(DOC, 6, "I")
    assert appendix_section_id(DOC, 6, " II ") == appendix_section_id(DOC, 6, "ii")


@pytest.mark.parametrize("section", ["", "A", "1", "IX!", "i i"])
def test_invalid_appendix_section_is_rejected(section: str) -> None:
    with pytest.raises(ValueError, match="Invalid appendix section"):
        appendix_section_id(DOC, 6, section)
    with pytest.raises(ValueError, match="Invalid appendix section"):
        appendix_section_clause_id(DOC, 6, section, 1)
    with pytest.raises(ValueError, match="Invalid appendix section"):
        appendix_section_point_id(DOC, 6, section, 1, "a")
