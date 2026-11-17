"""Tests for the code-side Có/Không verdict check (issue #195).

Ported from the planner's prototype (``gascomply-archive/probes/verdict_check_probe.py``):
these assert the exact comparison-word table, Vietnamese number format, polarity guard and
``n/a`` rules that prototype already measured, never a value tuned to make a test pass.
"""

from app.services import verdict

_INSURANCE_CLAUSE = "Cơ sở kinh doanh khí đốt có tổng lượng khí đốt tồn chứa trên 500 kg."
_AREA_CLAUSE = "Cửa hàng phải có diện tích tối thiểu là 12 m2."


# --- the parser: comparison words -----------------------------------------------------------


def test_strictly_greater_words():
    for clause in (
        "trên 500 kg",
        "vượt quá 500 kg",
        "lớn hơn 500 kg",
        "cao hơn 500 kg",
        "hơn 500 kg",
    ):
        assert verdict.thresholds(clause) == [(">", 500.0, "kg")]


def test_at_or_above_words():
    for clause in ("tối thiểu là 12 m2", "ít nhất 12 m2", "từ 12 m2 trở lên"):
        assert verdict.thresholds(clause) == [(">=", 12.0, "m2")]


def test_strictly_less_word_duoi():
    assert verdict.thresholds("dưới 200 m2") == [("<", 200.0, "m2")]


def test_bare_hon_in_the_greater_than_pattern_outranks_the_less_than_words_that_contain_it():
    """The bare word "hơn" outranks the longer "nhỏ/thấp/ít hơn" words that contain it.

    Each of "nhỏ hơn"/"thấp hơn"/"ít hơn" (the ``<`` tuple) ends in the bare word "hơn", which
    the ``>`` tuple also matches on its own, and the ``>`` tuple runs first (``COMPARATORS``'
    own order, ported unchanged from the prototype): its bare "hơn" alternative matches and
    consumes "hơn 200 m2" before the ``<`` tuple's own, longer alternative ever gets to try --
    so each of these three actually parses as ``>``, not ``<``. This is the prototype's own
    measured behaviour (issue #195's own trap: "do not tune the word lists to gold"), kept
    exactly as ported, not a bug this issue is in scope to fix.
    """
    for clause in ("nhỏ hơn 200 m2", "thấp hơn 200 m2", "ít hơn 200 m2"):
        assert verdict.thresholds(clause) == [(">", 200.0, "m2")]


def test_at_or_below_words():
    for clause in (
        "không quá 70 kg",
        "không vượt quá 70 kg",
        "không lớn hơn 70 kg",
        "không cao hơn 70 kg",
        "tối đa là 70 kg",
        "nhiều nhất 70 kg",
    ):
        assert verdict.thresholds(clause) == [("<=", 70.0, "kg")]


def test_negated_at_or_above_words():
    for clause in ("không dưới 3 m", "không nhỏ hơn 3 m", "không thấp hơn 3 m", "không ít hơn 3 m"):
        assert verdict.thresholds(clause) == [(">=", 3.0, "m")]


def test_negated_form_is_never_re_read_by_the_looser_positive_pattern():
    """The negated form "không vượt quá 70 kg" is ``<=``, never also ``>`` from a re-match."""
    assert verdict.thresholds("không vượt quá 70 kg") == [("<=", 70.0, "kg")]


# --- the parser: Vietnamese number format ----------------------------------------------------


def test_thousands_separator_is_not_a_decimal_point():
    """The thousands separator "1.000 kg" is 1000 kg, not 1 kg (the prototype's own first bug)."""
    assert verdict.thresholds("từ 1.000 kg trở lên") == [(">=", 1000.0, "kg")]


def test_comma_is_the_decimal_point():
    assert verdict.thresholds("trên 2,5 tấn") == [(">", 2.5, "tấn")]


def test_superscript_units_are_normalized():
    assert verdict.thresholds("tối thiểu là 12 m²") == [(">=", 12.0, "m2")]
    assert verdict.thresholds("không quá 5 m³") == [("<=", 5.0, "m3")]


def test_quantities_parses_the_same_number_and_unit_shapes():
    assert verdict.quantities("Cửa hàng chứa 300 kg khí đốt") == [(300.0, "kg")]
    assert verdict.quantities("Cửa hàng có diện tích 1.000 m2") == [(1000.0, "m2")]


# --- the opening verdict -----------------------------------------------------------------------


def test_opening_verdict_reads_co_khong_chua():
    assert verdict.opening_verdict("Có, phải mua bảo hiểm.") == "Y"
    assert verdict.opening_verdict("Không, không phải mua.") == "N"
    assert verdict.opening_verdict("Chưa đủ điều kiện.") == "N"


def test_opening_verdict_is_none_for_anything_else():
    assert verdict.opening_verdict("Chỉ khi cơ sở đáp ứng điều kiện X.") is None
    assert verdict.opening_verdict("Cửa hàng 300 kg có phải mua bảo hiểm không?") is None


# --- the polarity guard -------------------------------------------------------------------------


def test_polarity_words_force_n_a():
    question = "Cửa hàng 10 m2 có vi phạm quy định về diện tích không?"
    assert verdict.check(question, [_AREA_CLAUSE], "Có, vi phạm.") == "n/a"


def test_no_polarity_word_is_checked_normally():
    question = "Cửa hàng 10 m2 có đáp ứng yêu cầu diện tích không?"
    # 10 m2 does not satisfy ">= 12", so an opening "Có" contradicts the clause.
    assert verdict.check(question, [_AREA_CLAUSE], "Có, đáp ứng yêu cầu.") == "contradiction"
    assert verdict.check(question, [_AREA_CLAUSE], "Không, không đáp ứng yêu cầu.") == "ok"


# --- check(): the verbatim PR #194 strings ------------------------------------------------------


def test_check_flags_the_branchs_wrong_300kg_answer():
    question = "Cửa hàng gas chứa 300 kg khí có phải mua bảo hiểm cháy, nổ bắt buộc không?"
    answer = (
        "Có, cửa hàng gas chứa 300 kg khí phải mua bảo hiểm cháy, nổ bắt buộc vì lượng khí đốt "
        "tồn chứa vượt quá ngưỡng 500 kg…"
    )
    assert verdict.check(question, [_INSURANCE_CLAUSE], answer) == "contradiction"


def test_check_passes_mains_correct_300kg_answer():
    question = "Cửa hàng gas chứa 300 kg khí có phải mua bảo hiểm cháy, nổ bắt buộc không?"
    answer = "Không, cửa hàng gas chứa 300 kg khí không phải mua…"
    assert verdict.check(question, [_INSURANCE_CLAUSE], answer) == "ok"


# --- n/a: more than one quantity/threshold, or none at all --------------------------------------


def test_no_quantity_in_question_is_n_a():
    assert (
        verdict.check("Cơ sở có phải mua bảo hiểm không?", [_INSURANCE_CLAUSE], "Có, phải mua.")
        == "n/a"
    )


def test_two_quantities_in_question_is_n_a():
    question = "Cửa hàng 300 kg hoặc 600 kg có phải mua bảo hiểm không?"
    assert verdict.check(question, [_INSURANCE_CLAUSE], "Có, phải mua.") == "n/a"


def test_a_different_unit_threshold_among_the_clauses_is_ignored():
    """Only the one same-unit threshold (kg) is used; the m2 one in another clause is not."""
    question = "Cửa hàng 300 kg có phải mua bảo hiểm không?"
    clauses = [_INSURANCE_CLAUSE, "Diện tích tối thiểu là 12 m2."]
    assert verdict.check(question, clauses, "Không, không phải mua.") == "ok"


def test_two_same_unit_thresholds_is_n_a():
    question = "Cửa hàng 300 kg có phải mua bảo hiểm không?"
    clauses = [_INSURANCE_CLAUSE, "Tồn chứa không quá 1.000 kg."]
    assert verdict.check(question, clauses, "Có, phải mua.") == "n/a"


def test_no_threshold_in_clause_is_n_a():
    question = "Cửa hàng 300 kg có phải mua bảo hiểm không?"
    assert verdict.check(question, ["Cơ sở phải tuân thủ quy định PCCC."], "Có, phải mua.") == "n/a"


def test_no_opening_verdict_is_n_a():
    question = "Cửa hàng 300 kg có phải mua bảo hiểm không?"
    assert (
        verdict.check(question, [_INSURANCE_CLAUSE], "Chỉ khi cơ sở thuộc diện quản lý.") == "n/a"
    )


# --- evaluate(): the facts behind a contradiction, for the generator's log line -----------------


def test_evaluate_returns_facts_only_on_contradiction():
    question = "Cửa hàng gas chứa 300 kg khí có phải mua bảo hiểm cháy, nổ bắt buộc không?"
    answer = "Có, cửa hàng gas chứa 300 kg khí phải mua bảo hiểm cháy, nổ bắt buộc."

    result, facts = verdict.evaluate(question, [_INSURANCE_CLAUSE], answer)

    assert result == "contradiction"
    assert facts == verdict.VerdictFacts(
        quantity=300.0, unit="kg", op=">", threshold=500.0, expected="Không", opened="Có"
    )


def test_evaluate_returns_no_facts_on_ok_or_n_a():
    question = "Cửa hàng gas chứa 300 kg khí có phải mua bảo hiểm cháy, nổ bắt buộc không?"
    ok_result, ok_facts = verdict.evaluate(
        question, [_INSURANCE_CLAUSE], "Không, cửa hàng gas chứa 300 kg khí không phải mua."
    )
    na_result, na_facts = verdict.evaluate(
        question, [_INSURANCE_CLAUSE], "Chỉ khi thuộc diện quản lý."
    )

    assert (ok_result, ok_facts) == ("ok", None)
    assert (na_result, na_facts) == ("n/a", None)
