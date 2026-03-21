#!/usr/bin/env python3
"""
proposal_ranker.py
==================
Ranking module for R&D proposals.

Compares two or more JSON metadata files (produced by extraction_engine.py)
and determines which proposal is superior based on Ministry of Coal criteria.

Scoring Weights
---------------
- Budget        : 30 %  (favour proposals ≤ ₹50,00,000)
- Keyword Match : 50 %  (+5 per priority keyword, capped at 50)
- Timeline      : 20 %  (favour ≤ 24 months, penalise > 36 months)
- PI Bonus      : +3 for Dr./Prof., +1 for Mr./Ms./Mrs.

Usage
-----
    python proposal_ranker.py metadata1.json metadata2.json [metadata3.json ...]

Dependencies: rich
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
from typing import Any

# ── Force UTF-8 stdout on Windows to avoid cp1252 encoding errors ────────────
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── Configuration ────────────────────────────────────────────────────────────

#: Ministry of Coal priority keywords — each match adds +5 to the keyword score.
PRIORITY_KEYWORDS: list[str] = [
    "Methane",
    "Safety",
    "Carbon Capture",
    "Automation",
    "Clean Coal",
    "Emissions",
]

BUDGET_THRESHOLD_LOW: float = 50_00_000       # ₹50 Lakh — ideal ceiling
BUDGET_THRESHOLD_HIGH: float = 1_00_00_000    # ₹1 Crore — upper comfort zone

TIMELINE_IDEAL_MONTHS: int = 24
TIMELINE_PENALTY_MONTHS: int = 36

console = Console(force_terminal=True)

# ── Parsing Helpers ──────────────────────────────────────────────────────────


def parse_budget(budget_str: str) -> float:
    """
    Convert a budget string like '₹90,00,000' or 'Rs 45,00,000' to a float.

    Returns 0.0 if the string cannot be parsed.
    """
    if not budget_str or budget_str.lower() == "not detected":
        return 0.0

    # Strip currency symbols and whitespace
    cleaned = re.sub(r"[₹$]|Rs\.?|INR", "", budget_str, flags=re.IGNORECASE).strip()
    # Remove commas and spaces inside the number
    cleaned = cleaned.replace(",", "").replace(" ", "")

    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_timeline(timeline_str: str) -> int:
    """
    Extract duration in months from strings like '36 months (3 years)' or '2 years'.

    Returns 0 if the string cannot be parsed.
    """
    if not timeline_str or timeline_str.lower() == "not detected":
        return 0

    # Try months first
    m = re.search(r"(\d+)\s*months?", timeline_str, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Fall back to years
    y = re.search(r"(\d+)\s*years?", timeline_str, re.IGNORECASE)
    if y:
        return int(y.group(1)) * 12

    return 0


def academic_rank_bonus(pi_name: str) -> tuple[int, str]:
    """
    Return (bonus_points, rank_label) based on the PI's honorific.

    +3 for Dr. / Prof.
    +1 for Mr. / Ms. / Mrs.
     0 otherwise.
    """
    if not pi_name:
        return 0, "Unknown"

    name_lower = pi_name.lower().strip()

    if name_lower.startswith("dr.") or name_lower.startswith("prof.") or name_lower.startswith("professor"):
        return 3, "Dr./Prof."
    if name_lower.startswith("mr.") or name_lower.startswith("ms.") or name_lower.startswith("mrs."):
        return 1, "Mr./Ms./Mrs."

    return 0, "—"


# ── Scoring Engine ───────────────────────────────────────────────────────────


def calculate_score(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Score a single proposal and return a detailed breakdown.

    Returns
    -------
    dict with keys:
        title, pi, budget_raw, timeline_raw,
        budget_value, timeline_months,
        budget_score, keyword_score, timeline_score, pi_bonus, pi_rank,
        matched_keywords, total_score
    """
    # ── Budget (max 30) ──────────────────────────────────────────────────
    budget_value = parse_budget(metadata.get("budget", ""))

    if budget_value <= 0:
        budget_score = 15.0  # unknown — neutral
    elif budget_value <= BUDGET_THRESHOLD_LOW:
        budget_score = 30.0
    elif budget_value <= BUDGET_THRESHOLD_HIGH:
        # Linear interpolation: 30 → 15 between 50 L and 1 Cr
        ratio = (budget_value - BUDGET_THRESHOLD_LOW) / (BUDGET_THRESHOLD_HIGH - BUDGET_THRESHOLD_LOW)
        budget_score = 30.0 - ratio * 15.0
    else:
        budget_score = 5.0

    # ── Keywords (max 50) ────────────────────────────────────────────────
    proposal_keywords: list[str] = metadata.get("keywords", [])
    if isinstance(proposal_keywords, str):
        proposal_keywords = []

    matched_keywords: list[str] = []
    for pk in PRIORITY_KEYWORDS:
        for kw in proposal_keywords:
            if pk.lower() == kw.lower():
                matched_keywords.append(pk)
                break

    keyword_score = min(len(matched_keywords) * 5, 50)

    # ── Timeline (max 20) ────────────────────────────────────────────────
    timeline_months = parse_timeline(metadata.get("timeline", ""))

    if timeline_months <= 0:
        timeline_score = 10.0  # unknown — neutral
    elif timeline_months <= TIMELINE_IDEAL_MONTHS:
        timeline_score = 20.0
    elif timeline_months <= TIMELINE_PENALTY_MONTHS:
        # Linear interpolation: 20 → 10 between 24 and 36 months
        ratio = (timeline_months - TIMELINE_IDEAL_MONTHS) / (TIMELINE_PENALTY_MONTHS - TIMELINE_IDEAL_MONTHS)
        timeline_score = 20.0 - ratio * 10.0
    else:
        timeline_score = 5.0

    # ── PI Bonus ─────────────────────────────────────────────────────────
    pi_name = metadata.get("principal_investigator", "")
    pi_bonus, pi_rank = academic_rank_bonus(pi_name)

    total = budget_score + keyword_score + timeline_score + pi_bonus

    return {
        "title": metadata.get("project_title", "Untitled"),
        "pi": pi_name,
        "budget_raw": metadata.get("budget", "N/A"),
        "timeline_raw": metadata.get("timeline", "N/A"),
        "budget_value": budget_value,
        "timeline_months": timeline_months,
        "budget_score": round(budget_score, 1),
        "keyword_score": keyword_score,
        "timeline_score": round(timeline_score, 1),
        "pi_bonus": pi_bonus,
        "pi_rank": pi_rank,
        "matched_keywords": matched_keywords,
        "total_score": round(total, 1),
    }


# ── Comparison & Output ─────────────────────────────────────────────────────


def print_ranked_table(scored: list[dict[str, Any]]) -> None:
    """Print a ranked table of all proposals, sorted by total score descending."""
    ranked = sorted(scored, key=lambda s: s["total_score"], reverse=True)

    table = Table(
        title="[>>] Proposal Ranking - Ministry of Coal Criteria",
        title_style="bold bright_cyan",
        show_lines=True,
        header_style="bold magenta",
    )
    table.add_column("Rank", justify="center", style="bold yellow", width=5)
    table.add_column("Proposal Title", style="cyan", max_width=40)
    table.add_column("PI", style="white", max_width=22)
    table.add_column("Budget\n(30)", justify="center", style="green")
    table.add_column("Keywords\n(50)", justify="center", style="green")
    table.add_column("Timeline\n(20)", justify="center", style="green")
    table.add_column("PI\nBonus", justify="center", style="green")
    table.add_column("Total", justify="center", style="bold bright_white")

    for idx, s in enumerate(ranked, start=1):
        table.add_row(
            str(idx),
            s["title"],
            s["pi"],
            str(s["budget_score"]),
            str(s["keyword_score"]),
            str(s["timeline_score"]),
            f"+{s['pi_bonus']}",
            f"[bold]{s['total_score']}[/bold]",
        )

    console.print()
    console.print(table)
    console.print()


def print_comparison(a: dict[str, Any], b: dict[str, Any]) -> None:
    """Print a side-by-side comparison table for two proposals."""
    table = Table(
        title="[<>] Head-to-Head Comparison",
        title_style="bold bright_yellow",
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold white", width=22)
    table.add_column(a["title"][:35], style="bright_green", max_width=35)
    table.add_column(b["title"][:35], style="bright_blue", max_width=35)

    rows = [
        ("PI", a["pi"], b["pi"]),
        ("PI Rank", a["pi_rank"], b["pi_rank"]),
        ("Budget", a["budget_raw"], b["budget_raw"]),
        ("Timeline", a["timeline_raw"], b["timeline_raw"]),
        ("Matched Keywords", ", ".join(a["matched_keywords"]) or "None", ", ".join(b["matched_keywords"]) or "None"),
        ("Budget Score /30", str(a["budget_score"]), str(b["budget_score"])),
        ("Keyword Score /50", str(a["keyword_score"]), str(b["keyword_score"])),
        ("Timeline Score /20", str(a["timeline_score"]), str(b["timeline_score"])),
        ("PI Bonus", f"+{a['pi_bonus']}", f"+{b['pi_bonus']}"),
        ("TOTAL", f"[bold]{a['total_score']}[/bold]", f"[bold]{b['total_score']}[/bold]"),
    ]
    for label, va, vb in rows:
        table.add_row(label, va, vb)

    console.print(table)
    console.print()


def print_recommendation(scored: list[dict[str, Any]]) -> None:
    """Print the final recommendation panel for the best proposal."""
    ranked = sorted(scored, key=lambda s: s["total_score"], reverse=True)
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None

    # Build reasoning
    reasons: list[str] = []

    if best["budget_score"] >= 25:
        reasons.append("budget is well within the ₹50 Lakh threshold")
    elif best["budget_score"] >= 15:
        reasons.append("budget is within the ₹1 Crore comfort zone")
    else:
        reasons.append("budget exceeds the ideal range but other factors compensate")

    kw_count = len(best["matched_keywords"])
    if kw_count >= 4:
        reasons.append(f"strong alignment with {kw_count} Ministry priority keywords")
    elif kw_count >= 2:
        reasons.append(f"moderate alignment with {kw_count} priority keywords")
    else:
        reasons.append(f"limited keyword alignment ({kw_count} match)")

    if best["timeline_months"] and best["timeline_months"] <= TIMELINE_IDEAL_MONTHS:
        reasons.append(f"timeline of {best['timeline_months']} months is within the ideal 24-month window")
    elif best["timeline_months"] and best["timeline_months"] <= TIMELINE_PENALTY_MONTHS:
        reasons.append(f"timeline of {best['timeline_months']} months is acceptable")
    elif best["timeline_months"]:
        reasons.append(f"timeline of {best['timeline_months']} months is long but offset by other strengths")

    if best["pi_bonus"] >= 3:
        reasons.append("PI holds a senior academic rank (Dr./Prof.)")

    reason_str = "; ".join(reasons) + "."

    margin = ""
    if runner_up:
        diff = round(best["total_score"] - runner_up["total_score"], 1)
        margin = f"  Margin of victory: [bold]+{diff} points[/bold] over the runner-up."

    text = (
        f"[bold bright_green]>>> WINNER:[/bold bright_green]  "
        f"[bold]{best['title']}[/bold]  (Score: {best['total_score']})\n\n"
        f"[bold]Reason:[/bold] This proposal is rated [bold]Better[/bold] because its {reason_str}\n"
        f"{margin}"
    )

    console.print(
        Panel(
            text,
            title="[*] Final Recommendation",
            title_align="left",
            border_style="bright_green",
            padding=(1, 2),
        )
    )


# ── CLI ──────────────────────────────────────────────────────────────────────


def load_metadata(path: str) -> dict[str, Any]:
    """Load and validate a JSON metadata file."""
    if not os.path.isfile(path):
        console.print(f"[red][ERROR][/red] File not found: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as fp:
        try:
            data = json.load(fp)
        except json.JSONDecodeError as exc:
            console.print(f"[red][ERROR][/red] Invalid JSON in {path}: {exc}")
            sys.exit(1)

    return data


def main() -> None:
    files = sys.argv[1:]

    if len(files) < 2:
        console.print(
            Panel(
                "[bold]Usage:[/bold]  python proposal_ranker.py  file1.json  file2.json  [file3.json ...]\n\n"
                "Provide [bold]two or more[/bold] JSON metadata files to compare.",
                title="Proposal Ranker",
                border_style="yellow",
            )
        )
        sys.exit(1)

    console.print()
    console.rule("[bold bright_cyan]Proposal Ranker — Ministry of Coal[/bold bright_cyan]")
    console.print()

    # Score each proposal
    scored: list[dict[str, Any]] = []
    for f in files:
        console.print(f"  [*] Loading [cyan]{os.path.basename(f)}[/cyan] ...")
        meta = load_metadata(f)
        result = calculate_score(meta)
        scored.append(result)

    console.print()

    # 1. Ranked table
    print_ranked_table(scored)

    # 2. Side-by-side comparison (top 2)
    ranked = sorted(scored, key=lambda s: s["total_score"], reverse=True)
    print_comparison(ranked[0], ranked[1])

    # 3. Recommendation
    print_recommendation(scored)


if __name__ == "__main__":
    main()
