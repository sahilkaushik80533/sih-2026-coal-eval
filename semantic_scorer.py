"""
semantic_scorer.py
==================
AI-driven Semantic Scoring Engine for Coal R&D Proposals.

Uses the **Google Gemini API** (``google-generativeai``) to evaluate proposal
text against a strict Ministry of Coal rubric and return structured JSON
scores that can auto-populate the evaluation form.

Dependencies
------------
- ``google-generativeai >= 0.5.0``   (pip install google-generativeai)

Configuration
-------------
The API key is read from ``st.secrets["gemini"]["api_key"]`` at call time.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ── Soft import — the app still works without the Gemini SDK ─────────────────
try:
    import google.generativeai as genai

    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


# ── Constants ────────────────────────────────────────────────────────────────

#: Maximum characters of proposal text to send.  Gemini 1.5 Flash supports
#: up to ~1 M tokens, but we cap at 30 000 chars (~8 000 tokens) to stay
#: well within free-tier limits and keep latency low.
MAX_TEXT_CHARS = 30_000

#: The system prompt that turns Gemini into a Ministry of Coal auditor.
SYSTEM_PROMPT = """\
You are a **Ministry of Coal – Technical Auditor**.  Your job is to evaluate
an R&D proposal for the Indian Coal sector.  Be rigorous, fair, and concise.

Evaluate the proposal on **exactly four** dimensions, each on a 1–10 scale:

1. **Technical Innovation** — Is the science sound?  Does it use novel
   methods, emerging tech (AI / IoT / sensors), or advanced materials?
2. **Economic Viability** — Is the budget realistic?  What is the potential
   ROI?  Will it attract industry co-investment?
3. **Environmental Sustainability** — Does it reduce carbon emissions,
   improve waste management, or protect ecosystems?
4. **Ministry Alignment** — Does it address a 2026 Ministry of Coal
   strategic priority (e.g. Coal Gasification, Blue Hydrogen, Mine Safety,
   Carbon Capture, Pit Lake Management)?

Return your response **only** as a JSON object in this exact schema:

```json
{
  "technical_innovation": <int 1-10>,
  "economic_viability": <int 1-10>,
  "environmental_sustainability": <int 1-10>,
  "ministry_alignment": <int 1-10>,
  "reasoning": "<2-4 sentence summary explaining the scores>"
}
```

Rules:
- Return ONLY the JSON object.  No markdown fences, no extra text.
- Choose a score of 5 if the evidence is unclear.
- Be strict: a score of 8+ requires strong, explicit evidence in the text.
"""


def _clean_and_truncate(text: str) -> str:
    """Collapse whitespace and truncate to ``MAX_TEXT_CHARS``."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + "\n\n[…text truncated for evaluation…]"
    return text


def _extract_json(raw: str) -> dict[str, Any]:
    """
    Extract the first JSON object from *raw* text.

    Handles cases where Gemini wraps the JSON in markdown code fences.
    """
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = raw.strip().rstrip("`")

    # Find the first { … } block
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in AI response:\n{raw[:500]}")

    return json.loads(m.group())


def _clamp(value: Any, lo: int = 1, hi: int = 10) -> int:
    """Coerce *value* to an int in [lo, hi]."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return 5  # safe default
    return max(lo, min(hi, v))


# ── Public API ───────────────────────────────────────────────────────────────

def is_available() -> bool:
    """Return True if the Gemini SDK is installed."""
    return _GEMINI_AVAILABLE


def analyze_proposal(
    text: str,
    api_key: str,
    *,
    model_name: str = "gemini-1.5-flash",
) -> dict[str, Any]:
    """
    Send *text* to Gemini and return structured scores.

    Parameters
    ----------
    text : str
        Full proposal text (will be cleaned and truncated).
    api_key : str
        Google Gemini API key.
    model_name : str
        Model to use (default: ``gemini-1.5-flash``).

    Returns
    -------
    dict
        Keys: ``technical_innovation``, ``economic_viability``,
        ``environmental_sustainability``, ``ministry_alignment`` (int 1–10),
        ``reasoning`` (str), ``model`` (str).

    Raises
    ------
    RuntimeError
        If the SDK is not installed or the API call fails.
    """
    if not _GEMINI_AVAILABLE:
        raise RuntimeError(
            "`google-generativeai` is not installed.\n"
            "Run: pip install google-generativeai"
        )

    genai.configure(api_key=api_key)

    cleaned = _clean_and_truncate(text)
    if len(cleaned) < 100:
        raise ValueError(
            "The extracted text is too short to evaluate. "
            "Ensure the PDF has readable content."
        )

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT,
    )

    response = model.generate_content(
        f"Evaluate the following R&D proposal:\n\n{cleaned}",
        generation_config=genai.GenerationConfig(
            temperature=0.2,       # low creativity → consistent scores
            max_output_tokens=512, # JSON should be tiny
        ),
    )

    raw_text = response.text
    parsed = _extract_json(raw_text)

    return {
        "technical_innovation": _clamp(parsed.get("technical_innovation")),
        "economic_viability": _clamp(parsed.get("economic_viability")),
        "environmental_sustainability": _clamp(parsed.get("environmental_sustainability")),
        "ministry_alignment": _clamp(parsed.get("ministry_alignment")),
        "reasoning": str(parsed.get("reasoning", "No reasoning provided.")),
        "model": model_name,
    }
