from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional

import numpy as np

COMMON_HEADINGS = {
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusions",
    "discussion/conclusions",
    "acknowledgment",
    "acknowledgments",
    "references",
    "keywords",
}


def normalize_extracted_text(raw_text: str) -> str:
    r"""Normalize raw text from PDF extraction by rejoining hyphenations, URLs, and formatting headings.

    Parameters
    ----------
    raw_text : str
        Raw string extracted from a PDF page or document.

    Returns
    -------
    str
        Cleaned, normalized text ready for sentence tokenization.

    Examples
    --------
    >>> normalize_extracted_text("Inter-\n nationalization is key.")
    'Internationalization is key.'

    >>> normalize_extracted_text("Visit https://synia.\n toolforge.org/ for details.")
    'Visit https://synia.toolforge.org/ for details.'

    >>> normalize_extracted_text("Abstract\nThis paper presents a method.")
    'Abstract. This paper presents a method.'
    """
    if not raw_text:
        return ""

    # 1. Unicode NFKC normalization
    normalized = unicodedata.normalize("NFKC", raw_text)

    # 2. De-hyphenation: rejoin words broken across line breaks (hyphen or soft-hyphen \ufffe)
    normalized = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", normalized)
    normalized = re.sub(r"(\w+)\ufffe\s*\n?\s*(\w+)", r"\1\2", normalized)
    normalized = normalized.replace("\ufffe", "")

    # 3. Rejoin URLs split across line breaks (e.g. 'http://scholia.\n toolforge.org')
    normalized = re.sub(
        r"(https?://[^\s]+)\.\s*\n\s*([a-zA-Z0-9_\-\.]+)", r"\1.\2", normalized
    )
    normalized = re.sub(
        r"(https?://[^\s]+)/\s*\n\s*([a-zA-Z0-9_\-\.]+)", r"\1/\2", normalized
    )

    # 4. Handle standalone section headings to prevent merging into subsequent sentences
    processed_lines: List[str] = []
    for line in normalized.splitlines():
        line_str = line.strip()
        if not line_str:
            continue
        clean_lower = line_str.rstrip(":").strip().lower()
        if clean_lower in COMMON_HEADINGS and not line_str.endswith(
            (".", "!", "?", ":")
        ):
            line_str = line_str + "."
        elif (
            len(line_str) < 30
            and line_str.isupper()
            and not line_str.endswith((".", "!", "?", ":", ","))
        ):
            line_str = line_str + "."
        processed_lines.append(line_str)

    return " ".join(processed_lines)


def merge_and_sort_lines(
    dt_polys: List[Any],
    rec_texts: List[str],
    rec_scores: Optional[List[float]] = None,
    y_tol_ratio: float = 0.5,
) -> List[str]:
    """Cluster detected text boxes on the same vertical baseline and sort left-to-right.

    Parameters
    ----------
    dt_polys : List[Any]
        Polygon bounding box corner points from DBNet.
    rec_texts : List[str]
        Recognized text strings for each box.
    rec_scores : Optional[List[float]]
        Confidence scores for each text prediction.
    y_tol_ratio : float
        Vertical tolerance fraction of line height for clustering adjacent words.

    Returns
    -------
    List[str]
        Top-to-bottom, left-to-right ordered and merged text lines.

    Examples
    --------
    >>> polys = [
    ...     [[100, 10], [150, 10], [150, 30], [100, 30]],
    ...     [[10, 10], [60, 10], [60, 30], [10, 30]]
    ... ]
    >>> texts = ["World", "Hello"]
    >>> merge_and_sort_lines(polys, texts)
    ['Hello World']
    """
    if not dt_polys or not rec_texts:
        return [str(t).strip() for t in rec_texts if str(t).strip()]

    items: List[Dict[str, Any]] = []
    scores = rec_scores if rec_scores is not None else [1.0] * len(rec_texts)
    for poly, text, score in zip(dt_polys, rec_texts, scores):
        cleaned_text = str(text).strip()
        if not cleaned_text:
            continue
        pts = np.array(poly)
        ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
        xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
        h = max(ymax - ymin, 1.0)
        yc = (ymin + ymax) / 2.0
        items.append(
            {
                "text": cleaned_text,
                "score": float(score),
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "h": h,
                "yc": yc,
            }
        )

    if not items:
        return []

    # Sort boxes primarily by vertical position
    items.sort(key=lambda item: item["yc"])

    # Cluster words into horizontal lines
    lines: List[List[Dict[str, Any]]] = []
    for item in items:
        placed = False
        for line in lines:
            line_yc = sum(w["yc"] for w in line) / len(line)
            line_h = sum(w["h"] for w in line) / len(line)
            if abs(item["yc"] - line_yc) <= line_h * y_tol_ratio:
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])

    # Sort words within each line from left to right
    merged: List[str] = []
    for line in lines:
        line.sort(key=lambda item: item["xmin"])
        line_str = " ".join(item["text"] for item in line).strip()
        if line_str:
            merged.append(line_str)

    return merged

