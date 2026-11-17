"""Tests for the destructive-fixture database-name guard (issue #64).

Pure function, no database needed -- and must never be skipped, unlike every
other database-adjacent test in this suite, since it is what stops the
mistake that wiped the shared dev database from recurring silently.
"""

import pytest
from conftest import _require_test_database


def test_a_database_named_with_the_test_suffix_is_accepted():
    _require_test_database("postgresql+asyncpg://postgres:postgres@localhost:5433/gascomply_test")


def test_a_test_suffixed_database_with_a_different_stem_is_also_accepted():
    _require_test_database(
        "postgresql+asyncpg://postgres:postgres@localhost:5433/gascomply_64_test"
    )


@pytest.mark.parametrize(
    "database_url",
    [
        "postgresql+asyncpg://postgres:postgres@localhost:5433/gascomply",
        "postgresql+asyncpg://postgres:postgres@localhost:5433/gascomply_test_db",
        "postgresql+asyncpg://postgres:postgres@localhost:5433/test_gascomply",
    ],
    ids=["the-dev-database-itself", "suffix-not-at-the-end", "test-as-a-prefix-not-a-suffix"],
)
def test_a_database_not_ending_in_test_is_rejected(database_url: str):
    with pytest.raises(ValueError, match="must end in '_test'"):
        _require_test_database(database_url)


def test_the_error_names_the_offending_database():
    # The exact database that issue #64 was wiped from -- pinned directly,
    # not just a generic non-"_test" example.
    with pytest.raises(ValueError, match="gascomply"):
        _require_test_database("postgresql+asyncpg://postgres:postgres@localhost:5433/gascomply")
