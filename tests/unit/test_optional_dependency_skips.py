"""An absent optional extra is a skip; anything else is still a failure.

The rule and the reasoning live in ``tests/_harness/optional_deps.py``; the hook
that applies it is in ``tests/conftest.py``. These tests hold both to their
narrow contract, because a hook that turns failures into skips would hide real
breakage if it were even slightly too broad.
"""
from __future__ import annotations

import pytest

from tests._harness.optional_deps import (
    absent_optional_dependency,
    optional_dependency_names,
)


def test_the_set_is_read_from_the_project_not_hand_written():
    names = optional_dependency_names()
    # Packages that really are optional extras.
    for package in ("together", "replicate", "pypdf", "feedparser", "staticmap"):
        assert package in names, package
    # Core requirements and the test tooling never qualify: their absence is a
    # real problem and must keep failing.
    for package in ("pytest", "httpx", "pydantic", "effgen"):
        assert package not in names, package


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ModuleNotFoundError: No module named 'together'", "together"),
        ("ModuleNotFoundError: No module named 'yt_dlp'", "yt_dlp"),
        ("AssertionError: tool failed: feedparser is required: pip install feedparser",
         "feedparser"),
        ("staticmap is not installed. Install it with: pip install staticmap",
         "staticmap"),
    ],
)
def test_the_two_shapes_that_mean_not_installed_are_recognised(text, expected):
    assert absent_optional_dependency(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "assert 1 == 2",
        "ModuleNotFoundError: No module named 'effgen.core.base'",
        "ModuleNotFoundError: No module named 'pydantic'",
        "AssertionError: the answer is required",
        "",
    ],
)
def test_an_ordinary_failure_is_left_alone(text):
    assert absent_optional_dependency(text) is None
