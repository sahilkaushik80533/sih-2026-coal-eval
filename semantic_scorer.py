"""
semantic_scorer.py
==================
AI-driven Semantic Scoring Engine for Coal R&D Proposals.

Uses the **Google Gemini API** (``google-genai``) to evaluate proposal
text against a strict Ministry of Coal rubric and return structured JSON
scores that can auto-populate the evaluation form.

Dependencies
------------
- ``google-genai >= 1.0.0``   (pip install google-genai)

Configuration
-------------
The API key is read from ``st.secrets["GEMINI_API_KEY"]`` at call time.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ── Soft import — the app still works without the Gemini SDK ─────────────────
try:
    from google import genai
    from google.genai import types as genai_types

    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


# ── Constants ────────────────────────────────────────────────────────────────

#: Maximum characters of proposal text to send.  Gemini 1.5 Pro supports
#: up to ~2 M tokens, but we cap at 30 000 chars (~8 000 tokens) to stay
#: well within free-tier limits and keep latency low.
MAX_TEXT_CHARS = 30_000

#: Model name — stable endpoint; avoid `-latest` aliases which may 404.
MODEL_NAME = "gemini-1.5-flash"

#: Fallback models to try if the primary model returns a 404.
FALLBACK_MODELS = ["gemini-2.0-flash", "gemini-pro"]

#: Safety settings — set all categories to BLOCK_NONE.
#: Coal mining terminology (explosives, blasting, hazardous gases, etc.)
#: triggers false positives on default safety filters.
#: Built as google.genai types.SafetySetting objects.
SAFETY_SETTINGS = None  # Constructed lazily; see _build_safety_settings()


def _build_safety_settings() -> list:
    """Return a list of ``genai_types.SafetySetting`` with BLOCK_NONE."""
    return [
        genai_types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE",
        ),
        genai_types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE",
        ),
        genai_types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE",
        ),
        genai_types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE",
        ),
    ]

#: The system prompt that turns Gemini into a Ministry of Coal Technical Auditor.
SYSTEM_PROMPT = """\
You are a **Ministry of Coal — Technical Auditor**.  Your job is to evaluate
an R&D proposal submitted for the Indian Coal sector.  Be rigorous, fair,
and concise.

Evaluate the proposal on **exactly three** dimensions, each on a 1–10 scale:

1. **innovation_score** — Novelty of approach, use of emerging technologies
   (AI / IoT / drones / sensors), advanced materials, or novel methods.
   8+ requires strong, explicit evidence of breakthrough innovation.
2. **feasibility_score** — Budget realism, timeline achievability, team
   capability, infrastructure readiness, and prior pilot work.
   8+ requires clear evidence of prior results or institutional partnerships.
3. **impact_score** — Potential impact on coal sector efficiency, safety,
   environmental sustainability, carbon reduction, or alignment with
   Ministry of Coal 2026 strategic priorities (e.g. Coal Gasification,
   Blue Hydrogen, Mine Safety, Carbon Capture, Pit Lake Management).
   8+ requires quantified impact projections or strong policy alignment.

Also provide a **technical_summary**: exactly 2 sentences summarising the
proposal's core strengths and primary risks.

Return your response **only** as a raw JSON object with this exact schema:

{
  "innovation_score": <int 1-10>,
  "feasibility_score": <int 1-10>,
  "impact_score": <int 1-10>,
  "technical_summary": "<exactly 2 sentences>"
}

Rules:
- Return ONLY the raw JSON object.  No markdown fences, no backticks, no
  extra text before or after the JSON.
- Choose a score of 5 if the evidence is unclear or ambiguous.
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
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Send *text* to Gemini and return structured scores.

    Parameters
    ----------
    text : str
        Full proposal text (will be cleaned and truncated).
    api_key : str
        Google Gemini API key.
    model_name : str, optional
        Model to use.  Defaults to ``MODEL_NAME`` (``gemini-1.5-pro``).

    Returns
    -------
    dict
        Keys: ``innovation_score``, ``feasibility_score``, ``impact_score``
        (int 1–10), ``technical_summary`` (str), ``model`` (str).

    Raises
    ------
    RuntimeError
        If the SDK is not installed or the API call fails.
    """
    if not _GEMINI_AVAILABLE:
        raise RuntimeError(
            "`google-genai` is not installed.\n"
            "Run: pip install google-genai"
        )

    _model = model_name or MODEL_NAME

    client = genai.Client(api_key=api_key)

    cleaned = _clean_and_truncate(text)
    if len(cleaned) < 100:
        raise ValueError(
            "The extracted text is too short to evaluate. "
            "Ensure the PDF has readable content."
        )

    # Build the ordered list of models to attempt
    models_to_try = [_model] + [m for m in FALLBACK_MODELS if m != _model]

    last_error = None
    used_model = _model
    raw_text = None

    for candidate in models_to_try:
        try:
            response = client.models.generate_content(
                model=candidate,
                contents=f"Evaluate the following R&D proposal:\n\n{cleaned}",
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,       # low creativity → consistent scores
                    max_output_tokens=512, # JSON should be tiny
                    safety_settings=_build_safety_settings(),
                ),
            )
            raw_text = response.text
            used_model = candidate
            break  # success — stop trying
        except Exception as exc:
            last_error = exc
            err_str = str(exc).lower()
            # Retry with next model only on 404 / "not found" errors
            if "404" in err_str or "not found" in err_str:
                continue
            # Any other error — don't retry, raise immediately
            raise RuntimeError(
                f"Gemini API error (model: {candidate}): {exc}"
            ) from exc

    if raw_text is None:
        raise RuntimeError(
            f"All Gemini models failed. Last error: {last_error}\n"
            f"Models attempted: {models_to_try}"
        )

    parsed = _extract_json(raw_text)

    return {
        "innovation_score": _clamp(parsed.get("innovation_score")),
        "feasibility_score": _clamp(parsed.get("feasibility_score")),
        "impact_score": _clamp(parsed.get("impact_score")),
        "technical_summary": str(
            parsed.get("technical_summary", "No summary provided.")
        ),
        "model": used_model,
    }
