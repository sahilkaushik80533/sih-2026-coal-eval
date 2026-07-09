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

import ast
import json
import re
from typing import Any

from data_processing import normalize_ai_score_keys

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
MODEL_NAME = "gemini-2.5-flash"

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

#: The system prompt that turns Gemini into a strict Ministry of Coal R&D Evaluator.
SYSTEM_PROMPT = """\
You are an expert Technical Evaluator for the Ministry of Coal R&D Division.
Your job is to strictly and critically analyze the following research proposal.
Do NOT give generic high scores. You must justify every point.

Evaluate the text against these strict criteria on a scale of 1 to 10
(where 1 is completely unviable and 10 is industry-defining):
1. Technical Innovation: Does it introduce novel methodology, or just repeat standard practices?
2. Economic Viability: Is the budget realistic and justified by the proposed outcomes?
3. Environmental Sustainability: Does it actively reduce carbon footprint or environmental impact?
4. Ministry Alignment: Does it directly serve coal sector safety, efficiency, or sustainability targets?

CRITICAL INSTRUCTION: You must return ONLY a strictly valid JSON object.
Do NOT wrap it in markdown. Do NOT add extra conversational text.
Use exactly these keys:
{{
    "Project Name": "Extract exact title or return 'Unknown'",
    "Sponsoring Agency": "Extract exact agency or return 'Unknown'",
    "Budget": "Extract budget value or return 'Unknown'",
    "Duration": "Extract duration or return 'Unknown'",
    "technical_innovation": <integer 1-10>,
    "economic_viability": <integer 1-10>,
    "environmental_sustainability": <integer 1-10>,
    "ministry_alignment": <integer 1-10>,
    "reasoning": "Provide a strict, professional 2-sentence justification for the scores given. Point out specific flaws or strengths."
}}
"""


def _clean_and_truncate(text: str) -> str:
    """Collapse whitespace and truncate to ``MAX_TEXT_CHARS``."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + "\n\n[…text truncated for evaluation…]"
    return text


def parse_ai_response(raw_text: str) -> dict[str, Any]:
    """
    Resilient JSON parser for AI responses.

    Handles common LLM output issues:
    - Strips markdown code fences (```json ... ```)
    - Removes trailing commas before } or ]
    - Falls back to ``ast.literal_eval`` for single-quoted JSON
    - Falls back to regex field extraction as a last resort
    """
    # Step 1: strip markdown fences
    clean = re.sub(r"```(?:json)?\n?", "", raw_text)
    clean = re.sub(r"```", "", clean)
    clean = clean.strip()

    # Step 2: remove trailing commas (e.g.  {"a": 1,})
    clean = re.sub(r",\s*([}\]])", r"\1", clean)

    # Step 3: try standard json.loads
    try:
        return json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        pass

    # Step 4: try ast.literal_eval (handles single-quoted dicts)
    try:
        result = ast.literal_eval(clean)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    # Step 5: regex fallback — extract known fields
    def _re_int(key: str) -> int:
        m = re.search(rf'"{key}"\s*:\s*(\d+)', clean)
        return int(m.group(1)) if m else 5

    def _re_str(key: str) -> str:
        m = re.search(rf'"{key}"\s*:\s*"(.*?)"', clean, re.DOTALL)
        return m.group(1) if m else ""

    return {
        "Project Name": _re_str("Project Name") or "Unknown",
        "Sponsoring Agency": _re_str("Sponsoring Agency") or "Unknown",
        "Budget": _re_str("Budget") or "Unknown",
        "Duration": _re_str("Duration") or "Unknown",
        "technical_innovation": _re_int("technical_innovation"),
        "economic_viability": _re_int("economic_viability"),
        "environmental_sustainability": _re_int("environmental_sustainability"),
        "ministry_alignment": _re_int("ministry_alignment"),
        "reasoning": _re_str("reasoning") or "No justification provided.",
    }


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
        Model to use.  Defaults to ``MODEL_NAME``.

    Returns
    -------
    dict
        Keys: ``technical_innovation``, ``economic_viability``,
        ``environmental_sustainability``, ``ministry_alignment``
        (int 1–10), ``reasoning`` (str), ``model`` (str), plus
        metadata fields ``Project Name``, ``Sponsoring Agency``,
        ``Budget``, ``Duration``.

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
            prompt_text = f"""
You are an expert Technical Evaluator for the Ministry of Coal R&D Division. Your job is to strictly and critically analyze the following research proposal. Do NOT give generic high scores. You must justify every point.

Evaluate the text against these strict criteria on a scale of 1 to 10 (where 1 is completely unviable and 10 is industry-defining):
1. Technical Innovation: Does it introduce novel methodology, or just repeat standard practices?
2. Economic Viability: Is the budget realistic and justified by the proposed outcomes?
3. Environmental Sustainability: Does it actively reduce carbon footprint or environmental impact?
4. Ministry Alignment: Does it directly serve coal sector safety, efficiency, or sustainability targets?

CRITICAL INSTRUCTION: You must return ONLY a strictly valid JSON object. Do NOT wrap it in markdown. Do NOT add extra conversational text. Use exactly these keys:
{{{{
    "Project Name": "Extract exact title or return 'Unknown'",
    "Sponsoring Agency": "Extract exact agency or return 'Unknown'",
    "Budget": "Extract budget value or return 'Unknown'",
    "Duration": "Extract duration or return 'Unknown'",
    "technical_innovation": <integer 1-10>,
    "economic_viability": <integer 1-10>,
    "environmental_sustainability": <integer 1-10>,
    "ministry_alignment": <integer 1-10>,
    "reasoning": "Provide a strict, professional 2-sentence justification for the scores given. Point out specific flaws or strengths."
}}}}

Proposal Text:
{cleaned}
"""
            response = client.models.generate_content(
                model=candidate,
                contents=prompt_text,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
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

    parsed = parse_ai_response(raw_text)

    scored = {
        "Project Name": str(parsed.get("Project Name", "Unknown")),
        "Sponsoring Agency": str(parsed.get("Sponsoring Agency", "Unknown")),
        "Budget": str(parsed.get("Budget", "Unknown")),
        "Duration": str(parsed.get("Duration", "Unknown")),
        "technical_innovation": _clamp(parsed.get("technical_innovation")),
        "economic_viability": _clamp(parsed.get("economic_viability")),
        "environmental_sustainability": _clamp(parsed.get("environmental_sustainability")),
        "ministry_alignment": _clamp(parsed.get("ministry_alignment")),
        "reasoning": str(
            parsed.get("reasoning", "No justification provided.")
        ),
        "model": used_model,
    }

    return normalize_ai_score_keys(scored)
