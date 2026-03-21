#!/usr/bin/env python3
"""
extraction_engine.py
====================
High-precision PDF Information Extraction (IE) module for research proposals.

Usage
-----
    python extraction_engine.py [path/to/proposal.pdf]

If no path is given, defaults to ``sample_proposal.pdf`` in the working dir.

Outputs
-------
- ``metadata.json`` — structured JSON with all extracted fields.
- A formatted summary printed to stdout.

Dependencies: PyMuPDF (fitz)
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

import fitz  # PyMuPDF


# ── Configuration ────────────────────────────────────────────────────────────

#: Domain-specific keywords to scan for (case-insensitive matching).
DOMAIN_KEYWORDS: list[str] = [
    "Methane",
    "Safety",
    "Excavation",
    "Automation",
    "Clean Coal",
    "Underground Mining",
    "IoT",
    "Sensor",
    "LiDAR",
    "Ventilation",
    "Anomaly Detection",
    "Edge Computing",
    "Coal",
    "Mining",
    "Emissions",
    "Prototyping",
    "Machine Learning",
    "Gas Detection",
    "Hazardous",
]

NOT_DETECTED = "Not Detected"


# ── Text Extraction ─────────────────────────────────────────────────────────

def extract_text(pdf_path: str) -> str:
    """Open *pdf_path* with PyMuPDF and return the concatenated text of all pages."""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    text_parts: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text_parts.append(page_text)

    return "\n".join(text_parts)


# ── Field Extractors ────────────────────────────────────────────────────────

def _extract_title(text: str) -> str:
    """
    Heuristic to find the project title.

    Strategy (in priority order):
    1. Look for an explicit "Title:" label.
    2. Take the first multi-word line that is ≥ 6 words (likely a heading).
    """
    # Strategy 1 – explicit label
    m = re.search(
        r"(?:Project\s+)?Title\s*:\s*(.+)",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    # Strategy 2 – first substantial text block (skip short lines like dates)
    for line in text.splitlines():
        line = line.strip()
        if len(line.split()) >= 6:
            return line

    return NOT_DETECTED


def _extract_pi(text: str) -> str:
    """
    Extract the Principal Investigator / Lead Researcher name.

    Looks for patterns like:
        Principal Investigator: Dr. Name
        PI: Dr. Name
        Lead Researcher: Name
        Submitted by: Name
    """
    patterns = [
        r"(?:Principal\s+Investigator|PI)\s*:\s*(.+)",
        r"Lead\s+Researcher\s*:\s*(.+)",
        r"Submitted\s+by\s*:\s*(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            # Clean trailing org info after a comma if it looks like an address
            name = m.group(1).strip()
            # Remove trailing comma-separated institution for cleanliness
            name = re.split(r",\s*(?:IIT|NIT|IISC|University|Institute|Dept)", name, flags=re.IGNORECASE)[0].strip()
            return name

    return NOT_DETECTED


def _extract_budget(text: str) -> str:
    """
    Extract total budget / financial amount.

    Recognises ₹, Rs, Rs., and INR prefixes as well as "Total Budget:" labels.
    Returns the first match with the currency prefix preserved.
    """
    # Pattern: ₹ or Rs or INR followed by a number (with optional commas/decimals)
    amount_pat = r"[₹]\s*[\d,]+(?:\.\d+)?|(?:Rs\.?|INR)\s*[\d,]+(?:\.\d+)?"

    # 1. Try label-based extraction first (highest confidence)
    label_pat = rf"(?:Total\s+(?:Budget|Cost|Project\s+Cost))\s*[:\-–]?\s*({amount_pat})"
    m = re.search(label_pat, text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2. Fallback: pick the largest amount mentioned
    amounts = re.findall(amount_pat, text)
    if amounts:
        # Return all unique amounts found, joined
        return amounts[0].strip()

    return NOT_DETECTED


def _extract_timeline(text: str) -> str:
    """
    Extract project duration / timeline.

    Looks for patterns like:
        Duration: 36 months
        Duration: 3 years
        36 months (3 years)
        Project Duration: 24 months
    """
    # Explicit label
    label_patterns = [
        r"(?:Project\s+)?Duration\s*[:\-–]\s*(.+?)(?:\n|$)",
        r"(?:Project\s+)?Timeline\s*[:\-–]\s*(.+?)(?:\n|$)",
    ]
    for pat in label_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # Unlabelled "X months" or "X years"
    m = re.search(r"(\d+\s*(?:months?|years?))", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return NOT_DETECTED


def _extract_keywords(text: str) -> list[str]:
    """
    Scan *text* for domain-specific technical keywords.

    Returns a sorted, deduplicated list of keywords found.
    """
    found: set[str] = set()
    for kw in DOMAIN_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE):
            found.add(kw)
    return sorted(found) if found else []


# ── Orchestrator ─────────────────────────────────────────────────────────────

def extract_metadata(pdf_path: str) -> dict[str, Any]:
    """
    Run the full extraction pipeline on *pdf_path*.

    Returns a dict ready for JSON serialisation:
    ```json
    {
      "source_file": "...",
      "project_title": "...",
      "principal_investigator": "...",
      "budget": "...",
      "timeline": "...",
      "keywords": ["...", "..."]
    }
    ```
    """
    text = extract_text(pdf_path)

    title = _extract_title(text)
    pi = _extract_pi(text)
    budget = _extract_budget(text)
    timeline = _extract_timeline(text)
    keywords = _extract_keywords(text)

    return {
        "source_file": os.path.basename(pdf_path),
        "project_title": title,
        "principal_investigator": pi,
        "budget": budget,
        "timeline": timeline,
        "keywords": keywords if keywords else NOT_DETECTED,
    }


# ── Output helpers ───────────────────────────────────────────────────────────

def save_json(data: dict[str, Any], output_path: str = "metadata.json") -> str:
    """Write *data* to a JSON file and return its absolute path."""
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)
    return os.path.abspath(output_path)


def print_summary(data: dict[str, Any]) -> None:
    """Print a human-readable summary to stdout."""
    # Ensure stdout can handle Unicode (₹ etc.) on Windows
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    width = 72
    print("=" * width)
    print("  PDF INFORMATION EXTRACTION - RESULTS")
    print("=" * width)
    print(f"  Source File  : {data['source_file']}")
    print(f"  Title        : {data['project_title']}")
    print(f"  PI           : {data['principal_investigator']}")
    print(f"  Budget       : {data['budget']}")
    print(f"  Timeline     : {data['timeline']}")
    kw = data["keywords"]
    if isinstance(kw, list):
        print(f"  Keywords     : {', '.join(kw)}")
    else:
        print(f"  Keywords     : {kw}")
    print("=" * width)


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample_proposal.pdf"

    if not os.path.isfile(pdf_path):
        print(f"[ERROR] File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Extracting metadata from: {pdf_path}")
    metadata = extract_metadata(pdf_path)

    json_path = save_json(metadata)
    print(f"[INFO] Metadata exported to   : {json_path}\n")

    print_summary(metadata)


if __name__ == "__main__":
    main()
