"""Unit tests for chutes_miner.api.util (e.g. semcomp)."""

import pytest

from chutes_miner.api.util import semcomp


# semcomp returns: -1 if input < target, 0 if equal, 1 if input > target


def test_semcomp_exact_match():
    assert semcomp("0.6.0", "0.6.0") == 0
    assert semcomp("0.3.61", "0.3.61") == 0


def test_semcomp_less_than():
    assert semcomp("0.5.4", "0.6.0") == -1
    assert semcomp("0.3.59", "0.3.61") == -1
    assert semcomp("0.0.0", "0.6.0") == -1


def test_semcomp_greater_than():
    assert semcomp("0.7.0", "0.6.0") == 1
    assert semcomp("0.3.65", "0.3.61") == 1


def test_semcomp_strips_rc_suffix():
    """RC suffix is stripped; comparison uses only X.Y.Z."""
    assert semcomp("0.6.0rc0", "0.6.0") == 0
    assert semcomp("0.6.0-rc0", "0.6.0") == 0


def test_semcomp_normalizes_dot_rc_then_strips():
    """Version strings like 0.6.0.rc0 are normalized and then stripped to 0.6.0."""
    assert semcomp("0.6.0.rc0", "0.6.0") == 0
    assert semcomp("0.4.0.rc2", "0.4.0") == 0
    assert semcomp("0.4.0.rc16", "0.4.0") == 0


def test_semcomp_rc_below_target():
    """RC versions below target compare as less."""
    assert semcomp("0.5.4.rc9", "0.6.0") == -1
    assert semcomp("0.3.60.rc1", "0.3.61") == -1


def test_semcomp_empty_or_none_treated_as_zero():
    """Empty/falsy input is treated as 0.0.0."""
    assert semcomp("", "0.6.0") == -1
    assert semcomp("", "0.0.0") == 0


def test_semcomp_invalid_prefix_defaults_to_zero():
    """Non-matching version string yields 0.0.0."""
    assert semcomp("v0.6.0", "0.6.0") == -1  # "v" prefix -> no match -> 0.0.0
    assert semcomp("garbage", "0.6.0") == -1
