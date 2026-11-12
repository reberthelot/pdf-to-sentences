from __future__ import annotations

import doctest

from src.app.utils import normalization
from src.app.utils.normalization import merge_and_sort_lines, normalize_extracted_text


def test_normalization_doctests():
    """Verify that all doctests in src.app.utils.normalization pass successfully."""
    results = doctest.testmod(normalization)
    assert results.failed == 0, f"{results.failed} doctests failed"


def test_dehyphenation_standard():
    """Test standard hyphenation at end of lines."""
    raw = "The deep-learning tech-\n nique achieved great results."
    cleaned = normalize_extracted_text(raw)
    assert "technique" in cleaned
    assert "deep-learning" in cleaned


def test_soft_hyphen_handling():
    """Test hidden Unicode soft-hyphens (\\ufffe) emitted by some PDF renderers."""
    raw = "Unex\ufffepected behavior was observed."
    cleaned = normalize_extracted_text(raw)
    assert "Unexpected" in cleaned


def test_split_url_rejoining():
    """Test rejoining split URLs across line breaks."""
    raw = "See documentation at https://synia.\n toolforge.org/ for details."
    cleaned = normalize_extracted_text(raw)
    assert "https://synia.toolforge.org/" in cleaned


def test_heading_punctuation():
    """Test that standalone section headings have periods appended to prevent sentence blending."""
    raw = "Abstract\nThis study explores neural networks."
    cleaned = normalize_extracted_text(raw)
    assert "Abstract." in cleaned


def test_merge_and_sort_lines():
    """Test geometric clustering into horizontal baselines and left-to-right sorting."""
    # Two words on the same vertical line, but input out of order
    polys = [
        [[120, 50], [200, 50], [200, 70], [120, 70]],  # xmin: 120
        [[20, 52], [100, 52], [100, 72], [20, 72]],   # xmin: 20
    ]
    texts = ["Right", "Left"]
    merged = merge_and_sort_lines(polys, texts)
    assert merged == ["Left Right"]

