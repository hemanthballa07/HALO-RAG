"""Sample limits used by experiment commands."""

import pytest

from src.utils.cli import resolve_sample_limit


@pytest.mark.parametrize("requested,dry_run,cap,configured,expected", [
    (2, True, 30, 1000, 2),
    (100, True, 30, 1000, 30),
    (None, True, 20, 1000, 20),
    (2, False, 30, 1000, 2),
    (None, False, 30, 1000, 1000),
    (None, False, 30, None, None),
])
def test_resolve_sample_limit(requested, dry_run, cap, configured, expected):
    assert resolve_sample_limit(requested, dry_run, cap, configured) == expected


@pytest.mark.parametrize("requested", [0, -1])
def test_resolve_sample_limit_rejects_nonpositive_values(requested):
    with pytest.raises(ValueError, match="--limit must be a positive integer"):
        resolve_sample_limit(requested, True, 30)
