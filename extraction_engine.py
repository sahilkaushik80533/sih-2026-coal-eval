#!/usr/bin/env python3
"""
extraction_engine.py
====================
High-precision PDF Information Extraction (IE) module for research proposals.

Supports **two extraction paths**:
  1. **Digital PDF** — native text via PyMuPDF (primary).
  2. **Scanned PDF / OCR fallback** — if the digital text layer has < 100
     characters, the engine automatically converts each page to a 300 DPI
     image and runs Tesseract OCR.

Usage
-----
    python extraction_engine.py [path/to/proposal.pdf]

If no path is given, defaults to ``sample_proposal.pdf`` in the working dir.

Outputs
-------
- ``metadata.json`` — structured JSON with all extracted fields.
- A formatted summary printed to stdout.

Dependencies
------------
- PyMuPDF (fitz)
- pytesseract  (requires **Tesseract-OCR** installed on the system PATH)
- pdf2image    (requires **Poppler** installed on the system PATH)
- Pillow

.. important::
   SYSTEM REQUIREMENT: Tesseract-OCR and Poppler must be installed and
   available on the system PATH for the OCR fallback to work.
   - Tesseract: https://github.com/tesseract-ocr/tesseract
   - Poppler:   https://github.com/oschwartz10612/poppler-windows (Windows)
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

import fitz  # PyMuPDF

# ── OCR dependencies (soft import — only needed for scanned PDFs) ────────────
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image

    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


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
    # ── 2026 Ministry of Coal Strategic Goals ─────────────────────────
    "Coal Gasification",
    "Blue Hydrogen",
    "Perovskite Solar",
    "Fault Prediction",
    "Fluoride Removal",
    "Mine Safety Monitoring",
    "Waste to Wealth",
    "Carbon Capture",
    "Pit Lake Management",
]

#: Indian coal grade classifications and technical coal types.
COAL_GRADES: list[str] = [
    "Grade A", "Grade B", "Grade C", "Grade D", "Grade E", "Grade F", "Grade G",
    "Semi-Coking", "Coking Coal", "Non-Coking", "Washery Grade",
    "Anthracite", "Bituminous", "Sub-Bituminous", "Lignite", "Peat",
    "Thermal Coal", "Metallurgical Coal", "Steam Coal",
]

NOT_DETECTED = "Not Detected"

#: If the digital text layer has fewer characters than this, trigger OCR.
#: Raised from 100 → 1000 to catch 'phantom text' (PDFs with only headers
#: or metadata in their text layer that would otherwise bypass OCR).
OCR_TEXT_THRESHOLD = 1000

#: Validation keywords that a genuine research-proposal PDF should contain.
#: Used as a secondary quality gate: even if the text is long enough, it must
#: contain at least ``VALIDATION_KEYWORD_MIN`` of these terms to be trusted.
VALIDATION_KEYWORDS: list[str] = [
    "budget",
    "proposal",
    "ministry",
    "project",
    "technical",
]
VALIDATION_KEYWORD_MIN = 2  # minimum matches required


def _count_validation_matches(text: str) -> int:
    """Count how many ``VALIDATION_KEYWORDS`` appear in *text* (case-insensitive)."""
    text_lower = text.lower()
    return sum(1 for kw in VALIDATION_KEYWORDS if kw in text_lower)


# ── Text Cleaning ───────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Standardise extracted text (applied to *both* digital and OCR output).

    - Collapse consecutive whitespace / blank lines.
    - Strip non-printable control characters.
    - Trim leading/trailing whitespace.
    """
    # Remove non-printable chars (keep newlines, tabs, spaces)
    text = re.sub(r"[^\S\n\t]+", " ", text)
    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── OCR Engine ──────────────────────────────────────────────────────────────

def perform_ocr(pdf_path: str, dpi: int = 300) -> str:
    """
    Convert each page of *pdf_path* to a high-resolution image and run
    Tesseract OCR on it.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    dpi : int
        Resolution for the page-to-image conversion (default 300).

    Returns
    -------
    str
        Concatenated OCR text from all pages.

    Raises
    ------
    pytesseract.TesseractNotFoundError
        If the Tesseract-OCR executable is not on the system PATH.
    RuntimeError
        If the OCR libraries (pytesseract / pdf2image / Pillow) are not
        installed.
    """
    if not _OCR_AVAILABLE:
        raise RuntimeError(
            "OCR libraries not installed. "
            "Run:  pip install pytesseract pdf2image Pillow\n"
            "Also ensure Tesseract-OCR and Poppler are on the system PATH."
        )

    # Convert PDF pages → PIL Image objects at the requested DPI
    images: list[Image.Image] = convert_from_path(pdf_path, dpi=dpi)

    ocr_parts: list[str] = []
    for idx, img in enumerate(images, start=1):
        page_text = pytesseract.image_to_string(img)
        if page_text:
            ocr_parts.append(page_text)

    return "\n".join(ocr_parts)


# ── Text Extraction (with OCR fallback) ─────────────────────────────────────

def extract_text(pdf_path: str) -> tuple[str, bool]:
    """
    Extract text from *pdf_path*.

    **Primary path:** PyMuPDF digital text extraction.
    **Fallback:** OCR is triggered when *either* condition is true:

    1. The digital text has fewer than ``OCR_TEXT_THRESHOLD`` characters
       (likely a scanned document or near-empty text layer).
    2. Fewer than ``VALIDATION_KEYWORD_MIN`` validation keywords are found
       (the text layer is probably phantom/header-only metadata).

    Returns
    -------
    (text, is_ocr) : tuple[str, bool]
        The cleaned text and a flag indicating whether OCR was used.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    basename = os.path.basename(pdf_path)

    # ── 1. Try digital text extraction (PyMuPDF) ────────────────────────
    text_parts: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text_parts.append(page_text)

    digital_text = "\n".join(text_parts)
    is_ocr = False

    # ── 2. Decide: trust the digital text or fall back to OCR? ──────────
    text_len = len(digital_text.strip())
    kw_matches = _count_validation_matches(digital_text)

    needs_ocr = False
    ocr_reason = ""

    if text_len < OCR_TEXT_THRESHOLD:
        needs_ocr = True
        ocr_reason = (
            f"Digital text too short ({text_len} chars < {OCR_TEXT_THRESHOLD} threshold)"
        )
    elif kw_matches < VALIDATION_KEYWORD_MIN:
        needs_ocr = True
        ocr_reason = (
            f"Low keyword density detected ({kw_matches}/{VALIDATION_KEYWORD_MIN} "
            f"validation keywords found), forcing OCR"
        )

    if needs_ocr:
        print(f"[OCR] {basename}: {ocr_reason}. Switching to OCR path...")
        try:
            digital_text = perform_ocr(pdf_path)
            is_ocr = True
            print(f"[OCR] {basename}: OCR extraction completed successfully.")
        except Exception as exc:
            # If OCR also fails, proceed with whatever digital text we have
            print(f"[OCR] {basename}: OCR failed ({exc}), using available digital text.")
    else:
        print(
            f"[DIGITAL] {basename}: Digital text layer accepted "
            f"({text_len} chars, {kw_matches} validation keywords matched)."
        )

    # ── 3. Standardise ──────────────────────────────────────────────────
    return clean_text(digital_text), is_ocr


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


def _extract_coal_grade(text: str) -> str:
    """
    Detect Indian coal grade or coal type from the text.

    Checks for explicit labels first ("Coal Grade: C"), then scans
    for any mention of known grades/types from ``COAL_GRADES``.
    If multiple are found, returns the first (highest-confidence) match.
    """
    # Strategy 1: explicit label ("Coal Grade: D", "Grade: Semi-Coking")
    label_pat = r"(?:Coal\s+)?Grade\s*[:\-–]\s*([A-G]|\w[\w\s-]{2,30})"
    m = re.search(label_pat, text, re.IGNORECASE)
    if m:
        grade = m.group(1).strip().rstrip(".,:;")
        # If it's a single letter, normalise to "Grade X" for clarity
        if len(grade) == 1 and grade.upper() in "ABCDEFG":
            return grade.upper()
        return grade.title()

    # Strategy 2: scan for known grades/types
    for grade in COAL_GRADES:
        if re.search(rf"\b{re.escape(grade)}\b", text, re.IGNORECASE):
            # Return the canonical form from the list
            return grade

    return NOT_DETECTED


def _extract_organization(text: str) -> str:
    """
    Extract the lead researcher's organization / institution.

    Looks for IIT, NIT, IISC, CSIR, CMPDI, CMPDIL, university names, etc.
    """
    patterns = [
        r"(?:Organization|Institution|Affiliation|Institute|University)\s*[:\-–]\s*(.+?)(?:\n|$)",
        r"Submitted\s+(?:from|by)\s*[:\-–]?\s*(?:Dr\.?\s+\w+\s*,?\s*)?(.+?)(?:\n|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".,:;")

    # Scan for known Indian research institutions
    inst_patterns = [
        r"(IIT\s+\w+)",
        r"(NIT\s+\w+)",
        r"(IISc\b[\w\s]*)",
        r"(CSIR[\-\s]+\w[\w\s]*)",
        r"(CMPDI(?:L)?)",
        r"(Indian\s+School\s+of\s+Mines)",
        r"(ISM\s+Dhanbad)",
        r"(Central\s+Mine\s+Planning[\w\s]*)",
        r"(\w+\s+University)",
        r"(\w+\s+Institute\s+of\s+Technology)",
    ]
    for pat in inst_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    return NOT_DETECTED


# ── ML-Ready Text Normalisation ─────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Deep-clean text for downstream ML training / NLP pipelines.

    Operations:
    - Convert to lowercase
    - Strip accents and special Unicode characters
    - Replace currency symbols with tokens (₹ → INR)
    - Collapse all whitespace
    - Remove non-alphanumeric characters (keep spaces, periods, hyphens)
    """
    import unicodedata

    # Normalise Unicode (NFD) and strip accent marks
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    text = text.lower()
    text = text.replace("₹", "INR ").replace("rs.", "INR ").replace("rs ", "INR ")
    text = re.sub(r"[^a-z0-9\s.\-,%]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
      "keywords": ["...", "..."],
      "extraction_method": "Digital Text Layer" | "OCR (Scanned Document)"
    }
    ```
    """
    text, is_ocr = extract_text(pdf_path)

    title = _extract_title(text)
    pi = _extract_pi(text)
    budget = _extract_budget(text)
    timeline = _extract_timeline(text)
    keywords = _extract_keywords(text)

    coal_grade = _extract_coal_grade(text)
    organization = _extract_organization(text)

    return {
        "source_file": os.path.basename(pdf_path),
        "project_title": title,
        "principal_investigator": pi,
        "organization": organization,
        "budget": budget,
        "timeline": timeline,
        "coal_grade": coal_grade,
        "keywords": keywords if keywords else NOT_DETECTED,
        "extraction_method": "OCR (Scanned Document)" if is_ocr else "Digital Text Layer",
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
    print(f"  Extraction   : {data.get('extraction_method', 'Digital Text Layer')}")
    print(f"  Title        : {data['project_title']}")
    print(f"  PI           : {data['principal_investigator']}")
    print(f"  Organization : {data.get('organization', 'Not Detected')}")
    print(f"  Budget       : {data['budget']}")
    print(f"  Timeline     : {data['timeline']}")
    print(f"  Coal Grade   : {data.get('coal_grade', 'Not Detected')}")
    kw = data["keywords"]
    if isinstance(kw, list):
        print(f"  Keywords     : {', '.join(kw)}")
    else:
        print(f"  Keywords     : {kw}")
    print("=" * width)


# ── CLI entry point ──────────────────────────────────────────────────────────

# ── Smart Extraction for Form Pre-Fill ───────────────────────────────────────

#: Maps detected coal grade strings → form selectbox values.
_GRADE_FORM_MAP: dict[str, str] = {
    "a": "A", "b": "B", "c": "C", "d": "D", "e": "E", "f": "F", "g": "G",
    "grade a": "A", "grade b": "B", "grade c": "C", "grade d": "D",
    "grade e": "E", "grade f": "F", "grade g": "G",
    "semi-coking": "Semi-Coking", "coking coal": "Coking", "coking": "Coking",
    "non-coking": "Washery", "washery grade": "Washery", "washery": "Washery",
    "anthracite": "A", "bituminous": "C", "sub-bituminous": "D",
    "lignite": "F", "peat": "G",
    "thermal coal": "D", "metallurgical coal": "Coking", "steam coal": "D",
}


def smart_extract_for_form(pdf_path: str) -> dict[str, Any]:
    """
    Run the full extraction pipeline and return a dict whose keys match
    the New Entry form fields in ``app.py``.

    This is the **Smart Extraction** entry point — it extracts, cleans,
    and maps all fields so the form can be pre-filled, minimising manual
    data entry.

    Returns
    -------
    dict with keys:
        proposal_name, coal_grade, location, evaluator, remarks,
        extraction_method, confidence_flags
    """
    metadata = extract_metadata(pdf_path)

    # ── Map coal_grade → form selectbox value ────────────────────────
    raw_grade = metadata.get("coal_grade", NOT_DETECTED)
    form_grade = _GRADE_FORM_MAP.get(raw_grade.lower(), None)
    if form_grade is None and raw_grade != NOT_DETECTED:
        # Partial match: check if any key is a substring
        for key, val in _GRADE_FORM_MAP.items():
            if key in raw_grade.lower():
                form_grade = val
                break
    form_grade = form_grade or "D"  # sensible default

    # ── Build confidence flags (which fields were actually detected) ──
    flags: dict[str, bool] = {
        "title": metadata["project_title"] != NOT_DETECTED,
        "pi": metadata["principal_investigator"] != NOT_DETECTED,
        "organization": metadata.get("organization", NOT_DETECTED) != NOT_DETECTED,
        "budget": metadata["budget"] != NOT_DETECTED,
        "timeline": metadata["timeline"] != NOT_DETECTED,
        "coal_grade": metadata.get("coal_grade", NOT_DETECTED) != NOT_DETECTED,
    }
    detected_count = sum(flags.values())
    total_fields = len(flags)

    # ── Compose the form-ready dict ──────────────────────────────────
    return {
        # Direct form fields
        "proposal_name": metadata["project_title"] if flags["title"] else "",
        "coal_grade": form_grade,
        "location": metadata.get("organization", "") if flags["organization"] else "",
        "evaluator": metadata["principal_investigator"] if flags["pi"] else "",
        "remarks": (
            f"Budget: {metadata['budget']} | "
            f"Timeline: {metadata['timeline']} | "
            f"Keywords: {', '.join(metadata['keywords']) if isinstance(metadata['keywords'], list) else metadata['keywords']}"
        ),
        # Metadata for display
        "extraction_method": metadata["extraction_method"],
        "raw_budget": metadata["budget"],
        "raw_timeline": metadata["timeline"],
        "raw_coal_grade": metadata.get("coal_grade", NOT_DETECTED),
        "raw_organization": metadata.get("organization", NOT_DETECTED),
        "keywords": metadata["keywords"],
        "confidence_flags": flags,
        "confidence_score": f"{detected_count}/{total_fields}",
    }


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
