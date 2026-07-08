"""
data_processing.py
==================
Centralised data-filtration, sorting, and DataFrame sanitisation utilities
for the Ministry of Coal R&D Proposal Evaluation System.

All raw-data manipulation that was previously inlined in app.py is now
collected here so that the UI layer only calls high-level helpers.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
#  UNIVERSAL DATAFRAME SANITIZER
# ═══════════════════════════════════════════════════════════════════════════════


def clean_dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of *df* that is safe to pass to ``st.dataframe()`` /
    ``st.data_editor()`` without triggering PyArrow ``ArrowTypeError``.

    Steps
    -----
    1. Copy the frame (never mutates the caller's data).
    2. Force every column header to ``str``.
    3. For every ``object``-dtype column, replace ``NaN`` / ``None`` with
       ``""`` and convert remaining cells to ``str``.
    """
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].apply(
                lambda x: "" if pd.isna(x) or x is None else str(x)
            )
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  SHEET DATA — COERCION & COLUMN FIXING
# ═══════════════════════════════════════════════════════════════════════════════


def coerce_sheet_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce mixed-type columns returned by Google Sheets so that downstream
    numeric operations and Streamlit display work without errors.

    * ``Total Score`` and ``Technical Innovation`` are cast to numeric.
    * All remaining ``object`` columns have NaN filled with ``""`` and are
      cast to ``str``.
    """
    out = df.copy()

    for col in ("Total Score", "Technical Innovation"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].fillna("").astype(str)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  FILTRATION & SORTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def sort_by_total_score(df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    """Sort *df* by ``Total Score`` descending (default) and reset the index."""
    if "Total Score" not in df.columns:
        return df
    return df.sort_values("Total Score", ascending=ascending).reset_index(drop=True)


def filter_by_search(df: pd.DataFrame, term: str) -> pd.DataFrame:
    """
    Return rows where *term* appears (case-insensitive) in **any** column.
    Returns the full frame when *term* is empty.
    """
    if not term:
        return df
    mask = (
        df.astype(str)
        .apply(lambda col: col.str.contains(term, case=False, na=False))
        .any(axis=1)
    )
    return df[mask]


def add_rank_column(df: pd.DataFrame) -> pd.DataFrame:
    """Insert a 1-based ``Rank`` column at position 0."""
    out = df.copy()
    out.insert(0, "Rank", range(1, len(out) + 1))
    return out


def rank_proposals(proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return *proposals* sorted by ``total_score`` descending."""
    return sorted(proposals, key=lambda s: s["total_score"], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATAFRAME BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════


def build_ranked_table(proposals_ranked: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the ranked-proposals table shown on the Dashboard tab."""
    return pd.DataFrame([
        {
            "Rank": idx,
            "Proposal Title": s["title"],
            "PI": s["pi"],
            "Budget": s["budget_raw"],
            "Timeline": s["timeline_raw"],
            "Budget Score (/30)": s["budget_score"],
            "Keyword Score (/50)": s["keyword_score"],
            "Timeline Score (/20)": s["timeline_score"],
            "PI Bonus": s["pi_bonus"],
            "Total Score": s["total_score"],
            "Justification": s.get("justification", ""),
        }
        for idx, s in enumerate(proposals_ranked, 1)
    ])


def build_comparison_table(
    a: dict[str, Any],
    b: dict[str, Any],
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    """Build the side-by-side comparison table for Compare Mode."""
    return pd.DataFrame({
        "Metric": [
            "PI", "PI Rank", "Budget", "Timeline",
            "Priority Keywords Matched",
            "Budget Score (/30)", "Keyword Score (/50)",
            "Timeline Score (/20)", "PI Bonus", "TOTAL SCORE",
        ],
        label_a: [
            a["pi"], a["pi_rank"], a["budget_raw"], a["timeline_raw"],
            ", ".join(a["matched_keywords"]) or "None",
            a["budget_score"], a["keyword_score"],
            a["timeline_score"], f"+{a['pi_bonus']}", a["total_score"],
        ],
        label_b: [
            b["pi"], b["pi_rank"], b["budget_raw"], b["timeline_raw"],
            ", ".join(b["matched_keywords"]) or "None",
            b["budget_score"], b["keyword_score"],
            b["timeline_score"], f"+{b['pi_bonus']}", b["total_score"],
        ],
    })


def apply_leaderboard_gradient(
    df: pd.DataFrame,
) -> "pd.io.formats.style.Styler":
    """
    Apply colour-gradient styling to the leaderboard dataframe.

    Re-casts ``Total Score`` and ``Technical Innovation`` back to numeric
    (they may have been stringified by the sanitizer) and then applies
    background gradients.
    """
    styled_df = df.copy()
    for col in ("Total Score", "Technical Innovation"):
        if col in styled_df.columns:
            styled_df[col] = pd.to_numeric(styled_df[col], errors="coerce").fillna(0)

    styler = styled_df.style
    if "Total Score" in styled_df.columns:
        styler = styler.background_gradient(
            subset=["Total Score"], cmap="RdYlGn", vmin=0, vmax=100,
        )
    if "Technical Innovation" in styled_df.columns:
        styler = styler.background_gradient(
            subset=["Technical Innovation"], cmap="YlOrRd", vmin=1, vmax=10,
        )
    return styler


# ═══════════════════════════════════════════════════════════════════════════════
#  FUZZY KEY NORMALISATION — PREVENTS KeyErrors FROM VARIABLE AI OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════


def normalize_json_keys(raw_data_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Map variable AI-returned keys to strict standard column names.

    Iterates each item in *raw_data_list* and, for every expected standard
    column, performs a case-insensitive lookup against a set of known
    synonyms.  Missing keys default to ``""``.
    """
    key_maps: dict[str, list[str]] = {
        "Project Name": [
            "project name", "title", "project title", "title of project",
            "name", "proposal name", "proposal title",
        ],
        "Sponsoring Agency": [
            "sponsoring agency", "agency", "sponsor", "funding agency",
            "funding body", "organisation", "organization",
        ],
        "Budget": [
            "budget", "cost", "amount", "funding amount", "total budget",
            "total cost", "project cost",
        ],
        "Duration": [
            "duration", "period", "time", "timeline", "project duration",
            "time period", "months",
        ],
    }

    standardized: list[dict[str, Any]] = []
    for item in raw_data_list:
        new_item: dict[str, Any] = {}
        for std_key, variations in key_maps.items():
            found_value = ""
            for k, v in item.items():
                if k.lower().strip() in variations:
                    found_value = v
                    break
            new_item[std_key] = found_value
        standardized.append(new_item)

    return standardized


def normalize_ai_score_keys(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Map variable AI scorer output keys to the standard keys expected by
    the UI layer, preventing ``KeyError`` when the model returns synonyms.

    Standard output keys
    --------------------
    ``technical_innovation``, ``economic_viability``,
    ``environmental_sustainability``, ``ministry_alignment``,
    ``reasoning``, ``model``.
    """
    score_key_maps: dict[str, list[str]] = {
        "technical_innovation": [
            "technical_innovation", "innovation_score", "innovation",
            "tech_innovation", "novelty_score", "novelty",
        ],
        "economic_viability": [
            "economic_viability", "feasibility_score", "feasibility",
            "economic_score", "viability_score", "budget_realism",
        ],
        "environmental_sustainability": [
            "environmental_sustainability", "impact_score", "impact",
            "environmental_score", "sustainability_score", "env_score",
        ],
        "ministry_alignment": [
            "ministry_alignment", "alignment_score", "alignment",
            "policy_alignment", "strategic_alignment",
        ],
        "reasoning": [
            "reasoning", "technical_summary", "summary", "justification",
            "rationale", "explanation",
        ],
    }

    normalized: dict[str, Any] = {}
    for std_key, variations in score_key_maps.items():
        found = None
        for k, v in raw.items():
            if k.lower().strip() in variations:
                found = v
                break
        if std_key == "reasoning":
            normalized[std_key] = str(found) if found is not None else "No summary provided."
        else:
            # Clamp numeric scores to 1–10
            try:
                normalized[std_key] = max(1, min(10, int(found)))
            except (TypeError, ValueError):
                normalized[std_key] = 5  # safe default

    # Preserve the model field if present
    normalized["model"] = raw.get("model", "unknown")

    return normalized
