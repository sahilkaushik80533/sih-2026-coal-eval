"""
app.py
======
Streamlit front-end for the SIH 2026 R&D Proposal Evaluation System.

Integrates:
  - extraction_engine.py  → PDF → structured metadata
  - proposal_ranker.py    → metadata → weighted scores

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Local module imports ─────────────────────────────────────────────────────
from extraction_engine import extract_metadata
from proposal_ranker import (
    calculate_score,
    parse_budget,
    PRIORITY_KEYWORDS,
    STRATEGIC_KEYWORDS_2026,
)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SIH 2026 · R&D Proposal Evaluator",
    page_icon="⚒️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Ministry of Coal dark-blue theme ────────────────────────────
st.markdown(
    """
    <style>
    /* ── Root variables ─────────────────────────────────────────────── */
    :root {
        --navy:        #0a1628;
        --navy-light:  #132038;
        --accent:      #1e88e5;
        --accent-glow: #42a5f5;
        --gold:        #ffc107;
        --surface:     #16213e;
        --text:        #e8eaf6;
        --muted:       #90a4ae;
    }

    /* ── Header banner ──────────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--accent);
    }
    .main-header h1 {
        color: #ffffff;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .main-header p {
        color: var(--muted);
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }

    /* ── Metric card overrides ──────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: var(--surface);
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border: 1px solid rgba(30, 136, 229, 0.25);
    }
    div[data-testid="stMetric"] label {
        color: var(--muted) !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* ── Sidebar branding ───────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff;
    }

    /* ── Table styling ──────────────────────────────────────────────── */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* ── Tab styling ────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
    }

    /* ── Score badge ────────────────────────────────────────────────── */
    .score-badge {
        display: inline-block;
        background: linear-gradient(135deg, var(--accent), var(--accent-glow));
        color: #ffffff;
        font-weight: 700;
        font-size: 1.5rem;
        padding: 0.4rem 1.2rem;
        border-radius: 8px;
        margin: 0.3rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helper: extract from uploaded bytes ──────────────────────────────────────

def extract_from_upload(uploaded_file) -> tuple[dict[str, Any], bool]:
    """Save an uploaded PDF to a temp file. Run extraction_engine on it.

    Returns
    -------
    (metadata, is_ocr) : tuple[dict, bool]
    """
    suffix = ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        metadata = extract_metadata(tmp_path)
        # Override source_file with original upload name
        metadata["source_file"] = uploaded_file.name
    finally:
        os.unlink(tmp_path)

    is_ocr = metadata.get("extraction_method", "").startswith("OCR")
    return metadata, is_ocr


# ── Helper: Plotly score breakdown chart ─────────────────────────────────────

def score_breakdown_chart(scored: dict[str, Any], height: int = 300) -> go.Figure:
    """Return a horizontal bar chart showing the score components."""
    categories = ["Budget (/30)", "Keywords (/50)", "Timeline (/20)", "PI Bonus"]
    values = [
        scored["budget_score"],
        scored["keyword_score"],
        scored["timeline_score"],
        scored["pi_bonus"],
    ]
    max_vals = [30, 50, 20, 5]
    colors = ["#1e88e5", "#43a047", "#fb8c00", "#8e24aa"]

    fig = go.Figure()

    # Background (max possible)
    fig.add_trace(
        go.Bar(
            y=categories,
            x=max_vals,
            orientation="h",
            marker_color="rgba(255,255,255,0.07)",
            name="Max",
            hoverinfo="skip",
        )
    )
    # Actual scores
    fig.add_trace(
        go.Bar(
            y=categories,
            x=values,
            orientation="h",
            marker_color=colors,
            name="Score",
            text=[f"{v}" for v in values],
            textposition="inside",
            textfont=dict(color="white", size=13, family="Arial Black"),
        )
    )

    fig.update_layout(
        barmode="overlay",
        height=height,
        margin=dict(l=0, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, 55],
        ),
        yaxis=dict(
            showgrid=False,
            autorange="reversed",
            tickfont=dict(size=12),
        ),
        showlegend=False,
    )
    return fig


# ── Helper: display a proposal detail card ───────────────────────────────────

def render_proposal_card(scored: dict[str, Any]) -> None:
    """Render a proposal's full metadata + score breakdown."""
    st.markdown(
        f'<div class="score-badge">{scored["total_score"]} pts</div>',
        unsafe_allow_html=True,
    )

    col_meta, col_chart = st.columns([1, 1.3])

    with col_meta:
        st.markdown("##### Metadata")
        st.markdown(f"**Title:** {scored['title']}")
        st.markdown(f"**PI:** {scored['pi']}  ({scored['pi_rank']})")
        st.markdown(f"**Budget:** {scored['budget_raw']}")
        st.markdown(f"**Timeline:** {scored['timeline_raw']}")
        kw_str = ", ".join(scored["matched_keywords"]) if scored["matched_keywords"] else "None"
        st.markdown(f"**Priority Keywords Matched:** {kw_str}")

    with col_chart:
        st.markdown("##### Score Breakdown")
        st.plotly_chart(score_breakdown_chart(scored), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚒️ Ministry of Coal")
    st.markdown("### R&D Proposal Evaluator")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Upload R&D Proposal PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload two or more PDF proposals to compare them.",
    )

    st.markdown("---")
    st.caption("SIH 2026 · Built with Streamlit")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>R&D Proposal Evaluation Dashboard</h1>
        <p>Smart India Hackathon 2026 — Ministry of Coal · Automated Scoring & Comparison</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Process uploads ──────────────────────────────────────────────────────────
if not uploaded_files:
    st.info(
        "👈  **Upload two or more PDF proposals** from the sidebar to get started.",
        icon="📄",
    )
    st.stop()

# Use session state to cache results so we don't re-extract on every rerun
if "scored_proposals" not in st.session_state:
    st.session_state.scored_proposals = []
    st.session_state.processed_names = set()

# Process any new uploads
new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_names]

if new_files:
    for uf in new_files:
        try:
            with st.status(f"Processing **{uf.name}** ...", expanded=True) as status:
                st.write("Attempting digital text extraction (PyMuPDF)...")
                metadata, is_ocr = extract_from_upload(uf)

                if is_ocr:
                    st.info(
                        "Digital text layer not detected. "
                        "Initializing OCR Engine for scanned document analysis...",
                        icon="🔍",
                    )

                st.write(f"Extraction method: **{metadata.get('extraction_method', 'Digital Text Layer')}**")
                st.write("Calculating score...")

                scored = calculate_score(metadata)
                # Preserve raw metadata
                scored["keywords_all"] = metadata.get("keywords", [])
                scored["source_file"] = metadata.get("source_file", uf.name)
                scored["extraction_method"] = metadata.get("extraction_method", "Digital Text Layer")

                # Append OCR source info to justification
                if is_ocr:
                    existing = scored.get("justification", "")
                    ocr_note = "Source: Scanned Image (OCR processed)"
                    scored["justification"] = f"{ocr_note}. {existing}" if existing else ocr_note

                st.session_state.scored_proposals.append(scored)
                st.session_state.processed_names.add(uf.name)
                status.update(label=f"**{uf.name}** — done!", state="complete")

        except RuntimeError as exc:
            # OCR libraries not installed
            st.error(
                f"**OCR Error for {uf.name}:** {exc}\n\n"
                "**How to fix:**\n"
                "1. Install Python packages: `pip install pytesseract pdf2image Pillow`\n"
                "2. Install [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract) "
                "and add it to your system PATH.\n"
                "3. Install [Poppler](https://github.com/oschwartz10612/poppler-windows) "
                "(Windows) and add it to your system PATH.",
                icon="⚠️",
            )
        except Exception as exc:
            # Catch pytesseract.TesseractNotFoundError and other unexpected errors
            err_name = type(exc).__name__
            if "TesseractNotFound" in err_name:
                st.error(
                    f"**Tesseract OCR not found** while processing **{uf.name}**.\n\n"
                    "The scanned PDF requires OCR, but Tesseract is not installed.\n\n"
                    "**How to fix:**\n"
                    "1. Download and install [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract).\n"
                    "2. Ensure the `tesseract` executable is on your system PATH.\n"
                    "3. Restart the Streamlit app.",
                    icon="⚠️",
                )
            else:
                st.error(f"Failed to process **{uf.name}**: {exc}", icon="❌")

# Handle file removals (user cleared some files)
current_names = {f.name for f in uploaded_files}
st.session_state.scored_proposals = [
    s for s in st.session_state.scored_proposals if s["source_file"] in current_names
]
st.session_state.processed_names = {
    n for n in st.session_state.processed_names if n in current_names
}

proposals: list[dict[str, Any]] = st.session_state.scored_proposals

if not proposals:
    st.info("Upload PDF proposals from the sidebar to begin evaluation.")
    st.stop()

# Sort by score descending
proposals_ranked = sorted(proposals, key=lambda s: s["total_score"], reverse=True)

# Build a lookup by title
title_map: dict[str, dict] = {s["title"]: s for s in proposals_ranked}
titles = list(title_map.keys())

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_leader, tab_compare = st.tabs(["📊  Leaderboard", "⚖️  Compare Mode"])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1: LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_leader:

    # ── Winner callout ───────────────────────────────────────────────────
    best_proposal = proposals_ranked[0]
    st.success(
        f"🏆 **#1 Ranked Proposal:** {best_proposal['title']}  "
        f"— Score: **{best_proposal['total_score']} pts**",
        icon="🏆",
    )
    st.markdown("")

    # ── Metric cards ─────────────────────────────────────────────────────
    best = proposals_ranked[0]
    budgets = [
        (s["title"], s["budget_value"])
        for s in proposals_ranked
        if s["budget_value"] > 0
    ]
    best_budget_title = min(budgets, key=lambda x: x[1])[0] if budgets else "N/A"
    best_budget_val = min(budgets, key=lambda x: x[1])[1] if budgets else 0

    best_timeline = min(
        (s for s in proposals_ranked if s["timeline_months"] > 0),
        key=lambda s: s["timeline_months"],
        default=None,
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            label="🏆 Top Score",
            value=f"{best['total_score']} pts",
            delta=best["title"][:40],
        )
    with m2:
        fmt_budget = f"₹{best_budget_val:,.0f}" if best_budget_val else "N/A"
        st.metric(
            label="💰 Best Budget (Lowest)",
            value=fmt_budget,
            delta=best_budget_title[:40],
        )
    with m3:
        tl_val = f"{best_timeline['timeline_months']} months" if best_timeline else "N/A"
        tl_name = best_timeline["title"][:40] if best_timeline else ""
        st.metric(label="⏱️ Shortest Timeline", value=tl_val, delta=tl_name)

    st.markdown("")

    # ── Ranked table ─────────────────────────────────────────────────────
    st.markdown("### Ranked Proposals")

    df = pd.DataFrame(
        [
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
        ]
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Total Score": st.column_config.NumberColumn(format="%.1f"),
            "Justification": st.column_config.TextColumn(width="large"),
        },
    )

    # ── CSV Export ────────────────────────────────────────────────────────
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Leaderboard as CSV",
        data=csv_data,
        file_name="proposal_leaderboard.csv",
        mime="text/csv",
    )

    # ── Detailed view ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Detailed Proposal View")

    selected = st.selectbox(
        "Select a proposal to inspect",
        titles,
        key="detail_select",
    )

    if selected:
        render_proposal_card(title_map[selected])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2: COMPARE MODE
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### Side-by-Side Comparison")

    if len(titles) < 2:
        st.warning("Upload at least **two** proposals to enable comparison.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            pick_a = st.selectbox("Proposal A", titles, index=0, key="cmp_a")
        with c2:
            default_b = 1 if len(titles) > 1 else 0
            pick_b = st.selectbox("Proposal B", titles, index=default_b, key="cmp_b")

        if pick_a == pick_b:
            st.info("Select two **different** proposals to compare.")
        else:
            a = title_map[pick_a]
            b = title_map[pick_b]

            # ── Comparison table ─────────────────────────────────────────
            cmp_df = pd.DataFrame(
                {
                    "Metric": [
                        "PI",
                        "PI Rank",
                        "Budget",
                        "Timeline",
                        "Priority Keywords Matched",
                        "Budget Score (/30)",
                        "Keyword Score (/50)",
                        "Timeline Score (/20)",
                        "PI Bonus",
                        "TOTAL SCORE",
                    ],
                    pick_a: [
                        a["pi"],
                        a["pi_rank"],
                        a["budget_raw"],
                        a["timeline_raw"],
                        ", ".join(a["matched_keywords"]) or "None",
                        a["budget_score"],
                        a["keyword_score"],
                        a["timeline_score"],
                        f"+{a['pi_bonus']}",
                        a["total_score"],
                    ],
                    pick_b: [
                        b["pi"],
                        b["pi_rank"],
                        b["budget_raw"],
                        b["timeline_raw"],
                        ", ".join(b["matched_keywords"]) or "None",
                        b["budget_score"],
                        b["keyword_score"],
                        b["timeline_score"],
                        f"+{b['pi_bonus']}",
                        b["total_score"],
                    ],
                }
            )
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)

            # ── Side-by-side charts ──────────────────────────────────────
            st.markdown("#### Score Breakdown")
            ch1, ch2 = st.columns(2)
            with ch1:
                st.markdown(f"**{pick_a}**")
                st.plotly_chart(
                    score_breakdown_chart(a, height=250),
                    use_container_width=True,
                    key="cmp_chart_a",
                )
            with ch2:
                st.markdown(f"**{pick_b}**")
                st.plotly_chart(
                    score_breakdown_chart(b, height=250),
                    use_container_width=True,
                    key="cmp_chart_b",
                )

            # ── Winner callout ───────────────────────────────────────────
            diff = round(a["total_score"] - b["total_score"], 1)
            if diff > 0:
                winner, margin = pick_a, diff
            elif diff < 0:
                winner, margin = pick_b, abs(diff)
            else:
                winner, margin = None, 0

            if winner:
                st.success(
                    f"**Recommendation:** *{winner}* is rated **Better** "
                    f"with a margin of **+{margin} points**.",
                    icon="🏆",
                )
            else:
                st.info("Both proposals scored equally — consider qualitative factors.")
