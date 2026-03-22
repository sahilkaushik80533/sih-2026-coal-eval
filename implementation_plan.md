# Proposal Ranking Module — Implementation Plan

Build `proposal_ranker.py` to compare R&D proposals from JSON metadata files using Ministry of Coal criteria, with weighted scoring and a rich terminal output.

## Proposed Changes

### Scoring & Ranking Engine

#### [NEW] [proposal_ranker.py](file:///c:/Users/Sahil%20kaushik/Documents/R%20and%20D/proposal_ranker.py)

**Core helpers:**

| Helper | Purpose |
|---|---|
| `parse_budget(budget_str)` | Strips `₹`, commas → returns `float`. Falls back to `0.0`. |
| `parse_timeline(timeline_str)` | Extracts months (converts years → months). Falls back to `0`. |
| `academic_rank_bonus(pi_name)` | Returns `+3` for *Dr./Prof.*, `+1` for *Mr./Ms./Mrs.*, `0` otherwise. |

**`calculate_score(metadata: dict) → dict`** — returns a breakdown dict:

| Component | Weight | Rule |
|---|---|---|
| Budget (30 pts) | 30 % | ≤ ₹50,00,000 → 30; ≤ ₹1,00,00,000 → linear 15–30; > ₹1 Cr → 5 |
| Keywords (50 pts) | 50 % | +5 per priority keyword (Methane, Safety, Carbon Capture, Automation, Clean Coal, Emissions), capped at 50 |
| Timeline (20 pts) | 20 % | ≤ 24 mo → 20; 25-36 mo → linear 10–19; > 36 mo → 5 |
| PI Bonus | — | +3 (Dr./Prof.) or +1 (Mr./Ms./Mrs.) added on top |

**`compare_proposals(scored: list[dict])`** — prints:
1. A **Ranked List** table (rank, title, score breakdown, total).
2. A **Side-by-Side Comparison** table for the top-2 proposals.
3. A **Final Recommendation** panel with the winner and reasoning.

**CLI:** `python proposal_ranker.py metadata1.json metadata2.json [...]`

Uses `rich` library for styled terminal tables and panels.

---

### Test Data

#### [NEW] [metadata_proposal_b.json](file:///c:/Users/Sahil%20kaushik/Documents/R%20and%20D/metadata_proposal_b.json)

A second sample JSON file with different values (lower budget, shorter timeline, fewer keywords, `Mr.` PI) so the comparison produces meaningful contrast.

---

### Dependencies

#### [MODIFY] [requirements.txt](file:///c:/Users/Sahil%20kaushik/Documents/R%20and%20D/requirements.txt)

Add `rich>=13.0.0`.

## Verification Plan

### Automated Tests

Run the ranker against the two JSON files and confirm it exits cleanly with expected output:

```
python proposal_ranker.py metadata.json metadata_proposal_b.json
```

Expected: a ranked table, comparison table, and recommendation panel are printed without errors.

### Edge-Case Spot-Checks

- Run with a single file → should print "Need at least 2 proposals" message.
- Run with no arguments → should show usage help.
