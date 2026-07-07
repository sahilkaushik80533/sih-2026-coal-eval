"""
app.py
======
Streamlit front-end for the Ministry of Coal R&D Proposal Evaluation System.

Integrates:
  - extraction_engine.py  → PDF → structured metadata
  - proposal_ranker.py    → metadata → weighted scores
  - Google Sheets         → persist results to Ministry database

Run:
    streamlit run app.py

Google Sheets Setup
-------------------
To enable data persistence, create a ``.streamlit/secrets.toml`` file with
your GCP service-account credentials and spreadsheet URL.  See the template
shipped alongside this project for details.  The app runs in **offline mode**
when credentials are not configured.

Data Caching Strategy
---------------------
- ``fetch_sheet_data()`` is decorated with ``@st.cache_data(ttl=30)`` so
  reads from Google Sheets are cached for 30 seconds, avoiding redundant
  API calls on every Streamlit rerun.
- When new data is submitted (via the Entry form or the Submit button),
  ``st.cache_data.clear()`` is called explicitly to force a fresh read on
  the next render, keeping the View tab in real-time sync.
"""

from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime
from typing import Any

import matplotlib                       # Required by Styler.background_gradient
import matplotlib.pyplot as plt          # noqa: F401 — ensures matplotlib is fully loaded
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Local module imports ─────────────────────────────────────────────────────
from extraction_engine import extract_metadata, smart_extract_for_form
from proposal_ranker import (
    calculate_score,
    parse_budget,
    PRIORITY_KEYWORDS,
)
import semantic_scorer

# ── Google Drive API (for PDF uploads — no temp files) ───────────────────────
try:
    from google.oauth2.service_account import Credentials as SACredentials
    from googleapiclient.discovery import build as build_service
    from googleapiclient.http import MediaIoBaseUpload

    _DRIVE_LIB_AVAILABLE = True
except ImportError:
    _DRIVE_LIB_AVAILABLE = False

# ── Google Sheets (soft import — works without credentials) ──────────────────
try:
    from streamlit_gsheets import GSheetsConnection

    _GSHEETS_LIB_AVAILABLE = True
except ImportError:
    _GSHEETS_LIB_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────────────────────
SERVICE_ACCOUNT_EMAIL = "sih-robot@sih-coal-eval.iam.gserviceaccount.com"
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]

#: Columns for the coal-evaluation records worksheet.
EVAL_COLUMNS: list[str] = [
    "Timestamp",
    "Proposal Name",
    "Coal Grade",
    "Ash %",
    "Moisture %",
    "Volatile Matter %",
    "Fixed Carbon %",
    "GCV (kcal/kg)",
    "Location / Mine",
    "Evaluator",
    "Technical Innovation",
    "Economic Viability",
    "Environmental Sustainability",
    "Ministry Alignment",
    "Total Score",
    "PDF Link",
]

#: Weights for evaluation scoring (must sum to 1.0).
SCORE_WEIGHTS: dict[str, float] = {
    "Technical Innovation": 0.30,
    "Economic Viability": 0.25,
    "Environmental Sustainability": 0.25,
    "Ministry Alignment": 0.20,
}

# ── Gemini SDK (soft import — app works without it) ──────────────────────────
try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


def _resolve_gemini_key() -> str:
    """
    Locate the Gemini API key from ``st.secrets``.

    Supports two layouts:
      1. ``GEMINI_API_KEY = "…"``            (top-level)
      2. ``[gemini]\n  api_key = "…"``       (nested table)
    Returns an empty string if neither is found.
    """
    # Top-level key (user-requested format)
    key = st.secrets.get("GEMINI_API_KEY", "")
    if key:
        return key
    # Nested table (legacy format)
    key = st.secrets.get("gemini", {}).get("api_key", "")
    return key


# ── Safety settings — BLOCK_NONE to avoid false flags on mining terms ────────
# Now constructed as google.genai types.SafetySetting objects at call time.


def get_semantic_analysis(text: str) -> dict:
    """
    Call the Gemini API as a *Ministry of Coal Technical Auditor* and return
    a structured evaluation of the proposal text.

    Uses ``gemini-1.5-pro`` via the new ``google.genai`` Client SDK
    with all safety categories set to ``BLOCK_NONE`` to prevent false flags
    on coal-mining terminology.

    Returns
    -------
    dict
        ``innovation_score``  (int 1–10)
        ``feasibility_score`` (int 1–10)
        ``impact_score``      (int 1–10)
        ``technical_summary`` (str, 2 sentences)
        ``model``             (str, model name used)

    Raises
    ------
    RuntimeError
        If the Gemini SDK is missing or the API key is absent.
    """
    import json, re  # local — already at module top but keep scope clear

    if not _GENAI_AVAILABLE:
        raise RuntimeError(
            "`google-genai` is not installed.\n"
            "Run: pip install google-genai"
        )

    api_key = _resolve_gemini_key()
    if not api_key:
        raise RuntimeError(
            "Gemini API key not configured.  Add one of the following "
            "to `.streamlit/secrets.toml`:\n\n"
            '  GEMINI_API_KEY = "your-key"\n'
            "or\n"
            '  [gemini]\n  api_key = "your-key"'
        )

    # Truncate to ~30 000 chars to stay within free-tier limits
    cleaned = re.sub(r"\s+", " ", text).strip()[:30_000]
    if len(cleaned) < 100:
        raise ValueError(
            "Extracted text is too short for meaningful analysis. "
            "Ensure the PDF has readable content."
        )

    RUBRIC_PROMPT = """\
You are a **Ministry of Coal — Technical Auditor** reviewing an Indian Coal R&D
Proposal.  Be rigorous, fair, and concise.

Evaluate the proposal and return ONLY a raw JSON object (no markdown fences,
no backticks, no extra text) with this exact schema:

{
  "innovation_score": <int 1-10>,
  "feasibility_score": <int 1-10>,
  "impact_score": <int 1-10>,
  "technical_summary": "<exactly 2 sentences summarising strengths and risks>"
}

Scoring guidelines:
- **innovation_score**: Novelty of approach, use of emerging tech (AI/IoT/
  drones/sensors), advanced materials or methods.  8+ requires strong,
  explicit evidence of breakthrough innovation.
- **feasibility_score**: Budget realism, timeline achievability, team
  capability, infrastructure readiness.  8+ requires clear evidence of
  prior results or institutional partnerships.
- **impact_score**: Potential impact on coal sector efficiency, safety,
  environmental sustainability, carbon reduction, or alignment with
  Ministry of Coal 2026 strategic priorities (Coal Gasification, Blue
  Hydrogen, Mine Safety, Carbon Capture, Pit Lake Management).  8+ requires
  quantified impact projections or strong policy alignment.

Rules:
- Return ONLY the raw JSON object — no markdown fences, no extra commentary.
- If evidence is ambiguous, default to 5.
- Be strict: 8+ requires strong, explicit evidence.
"""

    model_name = "gemini-1.5-pro"
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=f"Evaluate the following R&D proposal:\n\n{cleaned}",
        config=genai_types.GenerateContentConfig(
            system_instruction=RUBRIC_PROMPT,
            temperature=0.2,
            max_output_tokens=512,
            safety_settings=[
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE",
                ),
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE",
                ),
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE",
                ),
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE",
                ),
            ],
        ),
    )

    raw = response.text
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object in AI response:\n{raw[:500]}")
    parsed = json.loads(m.group())

    def _clamp(v, lo=1, hi=10):
        try:
            return max(lo, min(hi, int(v)))
        except (TypeError, ValueError):
            return 5

    return {
        "innovation_score": _clamp(parsed.get("innovation_score")),
        "feasibility_score": _clamp(parsed.get("feasibility_score")),
        "impact_score": _clamp(parsed.get("impact_score")),
        "technical_summary": str(parsed.get("technical_summary", "No summary provided.")),
        "model": model_name,
    }

def calculate_eval_score(
    innovation: int, economic: int, environmental: int, alignment: int,
) -> float:
    """Compute weighted total score (0–100) from four 1–10 sliders."""
    raw = (
        innovation * SCORE_WEIGHTS["Technical Innovation"]
        + economic * SCORE_WEIGHTS["Economic Viability"]
        + environmental * SCORE_WEIGHTS["Environmental Sustainability"]
        + alignment * SCORE_WEIGHTS["Ministry Alignment"]
    )
    # Normalise: sliders are 1–10, so max raw = 10, min raw = 1
    # Map to 0–100 scale: (raw - 1) / (10 - 1) * 100
    return round((raw - 1) / 9 * 100, 1)

#: Columns for the proposal-submission worksheet.
PROPOSAL_COLUMNS: list[str] = [
    "Timestamp", "Title", "PI", "Budget",
    "Timeline", "Total Score", "Justification",
]


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ministry of Coal · R&D Proposal Evaluator",
    page_icon="⚒️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS — Ministry of Coal dark-blue theme ────────────────────────────
st.markdown(
    """
    <style>
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
    .main-header {
        background: linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--accent);
    }
    .main-header h1 { color: #fff; margin: 0; font-size: 1.8rem; font-weight: 700; letter-spacing: .5px; }
    .main-header p  { color: var(--muted); margin: .3rem 0 0 0; font-size: .95rem; }
    div[data-testid="stMetric"] {
        background: var(--surface); padding: 1rem 1.2rem;
        border-radius: 10px; border: 1px solid rgba(30,136,229,.25);
    }
    div[data-testid="stMetric"] label {
        color: var(--muted) !important; font-size: .85rem !important;
        text-transform: uppercase; letter-spacing: 1px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #fff !important; font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 { color: #fff; }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0; padding: .6rem 1.4rem; font-weight: 600;
    }
    .score-badge {
        display: inline-block;
        background: linear-gradient(135deg, var(--accent), var(--accent-glow));
        color: #fff; font-weight: 700; font-size: 1.5rem;
        padding: .4rem 1.2rem; border-radius: 8px; margin: .3rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  GOOGLE SHEETS — CONNECTION & DATA OPS
# ═══════════════════════════════════════════════════════════════════════════════


def _validate_secrets() -> str | None:
    """
    Pre-flight check: make sure the critical secrets fields are filled in.

    Returns None if everything looks good, or an error message string.
    """
    try:
        gs = st.secrets["connections"]["gsheets"]
    except (KeyError, FileNotFoundError):
        return (
            "**Secrets not configured.** Create `.streamlit/secrets.toml` "
            "with your Google Sheets credentials.\n\n"
            "See the template shipped with this project for the exact format."
        )

    spreadsheet_url = gs.get("spreadsheet", "")
    private_key = gs.get("private_key", "")
    client_email = gs.get("client_email", "")

    missing = []
    if not spreadsheet_url:
        missing.append("`spreadsheet` (full Google Sheet URL)")
    if not private_key:
        missing.append("`private_key` (from your JSON key file)")
    if not client_email:
        missing.append("`client_email` (service account email)")

    if missing:
        items = "\n".join(f"  - {m}" for m in missing)
        return (
            f"**Incomplete credentials.** The following fields in "
            f"`.streamlit/secrets.toml` are empty:\n\n{items}\n\n"
            "Fill them in from your GCP service-account JSON key file."
        )
    return None


def _get_gsheets_connection():
    """
    Establish and validate a Google Sheets connection.

    Returns
    -------
    (conn, error_msg) : tuple
        conn is a GSheetsConnection or None; error_msg explains the failure.

    The function runs three checks:
      1. Library installed?
      2. Secrets filled in?  (catches the "File not found" for empty URLs)
      3. Validation ping — actually reads the sheet to catch 403 / 404 early.
    """
    if not _GSHEETS_LIB_AVAILABLE:
        return None, (
            "`st-gsheets-connection` is not installed.\n\n"
            "Run: `pip install st-gsheets-connection`"
        )

    # ── Step 1: Validate secrets before even trying to connect ────────
    secrets_err = _validate_secrets()
    if secrets_err:
        return None, secrets_err

    # ── Step 2: Create the connection object ──────────────────────────
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
    except Exception as exc:
        return None, f"Connection init failed: {exc}"

    # ── Step 3: Validation ping — try reading 0 rows ─────────────────
    #   This surfaces 403/404/FileNotFound immediately instead of letting
    #   them appear later in fetch_sheet_data() or append_row().
    try:
        conn.read(worksheet="Sheet1", usecols=[0], ttl=5)
    except FileNotFoundError:
        return None, (
            "**Spreadsheet not found.** The URL in `secrets.toml` is "
            "incorrect or the sheet has been deleted.\n\n"
            "Open `.streamlit/secrets.toml` and set `spreadsheet` to the "
            "full URL of your Google Sheet\n"
            "(e.g. `https://docs.google.com/spreadsheets/d/…/edit`)."
        )
    except PermissionError:
        return None, (
            "**Permission denied.** The service account does not have "
            "editor access to the Google Sheet.\n\n"
            "**Fix:** Open your Google Sheet → **Share** → add\n"
            f"`{SERVICE_ACCOUNT_EMAIL}` as **Editor**."
        )
    except Exception as exc:
        err_str = str(exc).lower()
        if "permission" in err_str or "403" in err_str:
            return None, (
                f"**Permission denied (403).** The service account "
                f"`{SERVICE_ACCOUNT_EMAIL}` does not have access.\n\n"
                "**Fix:** Open your Google Sheet → **Share** → add\n"
                f"`{SERVICE_ACCOUNT_EMAIL}` as **Editor**."
            )
        if "not found" in err_str or "404" in err_str:
            return None, (
                "**Sheet not found (404).** Verify the `spreadsheet` URL in "
                "`.streamlit/secrets.toml`."
            )
        if "file not found" in err_str or "no such file" in err_str:
            return None, (
                "**Spreadsheet not found.** The URL in `secrets.toml` "
                "appears to be invalid.\n\n"
                "Set `spreadsheet` to the full URL of your Google Sheet."
            )
        return None, f"Connection validation failed: {exc}"

    return conn, None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_sheet_data(_conn, worksheet: str = "Sheet1") -> pd.DataFrame | None:
    """
    Read all rows from *worksheet*.  Cached for 30 s to reduce API calls.

    Pass the connection object as ``_conn`` (underscore prefix tells
    Streamlit not to hash it).
    """
    try:
        data = _conn.read(worksheet=worksheet)
        if data is not None and not data.empty:
            data = data.dropna(how="all")
        return data
    except FileNotFoundError:
        st.error(
            "**Spreadsheet not found.** The URL in `secrets.toml` may be "
            "incorrect.\n\nVerify `spreadsheet` in `.streamlit/secrets.toml`.",
            icon="🔴",
        )
        return None
    except Exception as exc:
        err = str(exc).lower()
        if "permission" in err or "403" in err:
            st.error(
                f"**Permission denied.** Share the Google Sheet with "
                f"`{SERVICE_ACCOUNT_EMAIL}` as **Editor**.",
                icon="🔴",
            )
        else:
            st.error(f"Failed to read sheet: {exc}", icon="❌")
        return None


def append_row(conn, worksheet: str, columns: list[str],
               row_dict: dict[str, Any]) -> tuple[bool, str]:
    """
    Append *row_dict* as a new row to *worksheet*.

    Returns (success, message).
    """
    try:
        existing = conn.read(worksheet=worksheet)
        if existing is None or existing.empty:
            existing = pd.DataFrame(columns=columns)
        else:
            existing = existing.dropna(how="all")

        new_row = pd.DataFrame([row_dict])
        updated = pd.concat([existing, new_row], ignore_index=True)
        conn.update(worksheet=worksheet, data=updated)

        # Bust the cache so the View tab refreshes immediately
        st.cache_data.clear()
        return True, "Row appended successfully."
    except FileNotFoundError:
        return False, (
            "**Spreadsheet not found.** Verify the `spreadsheet` URL in "
            "`.streamlit/secrets.toml`."
        )
    except PermissionError:
        return False, (
            f"**Permission denied.** Share the sheet with "
            f"`{SERVICE_ACCOUNT_EMAIL}` as Editor."
        )
    except Exception as exc:
        err = str(exc).lower()
        if "permission" in err or "403" in err:
            return False, (
                f"**Permission denied (403).** Share the sheet with "
                f"`{SERVICE_ACCOUNT_EMAIL}` as Editor."
            )
        if "not found" in err or "404" in err:
            return False, (
                "**Sheet not found.** Verify the `spreadsheet` URL in "
                "`.streamlit/secrets.toml`."
            )
        return False, f"Write failed: {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
#  GOOGLE DRIVE — PDF UPLOAD (byte-streaming, no temp files)
# ═══════════════════════════════════════════════════════════════════════════════


def _build_drive_service():
    """
    Build a Google Drive API v3 service using the same service-account
    credentials stored in ``st.secrets``.

    Returns (service, error_msg).
    """
    if not _DRIVE_LIB_AVAILABLE:
        return None, (
            "`google-api-python-client` / `google-auth` not installed.\n\n"
            "Run: `pip install google-api-python-client google-auth`"
        )
    try:
        gs = st.secrets["connections"]["gsheets"]
        info = {
            "type": gs.get("type", "service_account"),
            "project_id": gs.get("project_id", ""),
            "private_key_id": gs.get("private_key_id", ""),
            "private_key": gs.get("private_key", ""),
            "client_email": gs.get("client_email", ""),
            "client_id": gs.get("client_id", ""),
            "auth_uri": gs.get("auth_uri", ""),
            "token_uri": gs.get("token_uri", ""),
            "auth_provider_x509_cert_url": gs.get("auth_provider_x509_cert_url", ""),
            "client_x509_cert_url": gs.get("client_x509_cert_url", ""),
        }
        creds = SACredentials.from_service_account_info(info, scopes=DRIVE_SCOPES)
        service = build_service("drive", "v3", credentials=creds)
        return service, None
    except Exception as exc:
        return None, f"Drive auth failed: {exc}"


def upload_pdf_to_drive(
    file_bytes: bytes,
    filename: str,
    folder_id: str,
) -> tuple[str | None, str]:
    """
    Upload *file_bytes* as a PDF to Google Drive *folder_id*.

    Streams bytes directly via ``MediaIoBaseUpload`` — **no temp files**.

    Returns
    -------
    (shareable_url | None, message)
    """
    service, err = _build_drive_service()
    if err:
        return None, err

    try:
        file_metadata = {
            "name": filename,
            "mimeType": "application/pdf",
            "parents": [folder_id],
        }
        media = MediaIoBaseUpload(
            io.BytesIO(file_bytes),
            mimetype="application/pdf",
            resumable=True,
        )
        uploaded = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id,webViewLink")
            .execute()
        )

        # Make the file viewable by anyone with the link
        service.permissions().create(
            fileId=uploaded["id"],
            body={"role": "reader", "type": "anyone"},
        ).execute()

        link = uploaded.get("webViewLink", f"https://drive.google.com/file/d/{uploaded['id']}/view")
        return link, f"Uploaded **{filename}** to Google Drive."

    except Exception as exc:
        err_str = str(exc).lower()
        if "permission" in err_str or "403" in err_str:
            return None, (
                f"**Drive permission denied.** Share the target folder with "
                f"`{SERVICE_ACCOUNT_EMAIL}` as **Editor**, or check that "
                "the Google Drive API is enabled in your GCP project."
            )
        if "not found" in err_str or "404" in err_str:
            return None, (
                "**Drive folder not found (404).** Verify `drive_folder_id` "
                "in `.streamlit/secrets.toml`."
            )
        return None, f"Drive upload failed: {exc}"


# Legacy wrapper used by the Dashboard tab's "Submit to Ministry" button.
def submit_to_sheets(conn, scored: dict[str, Any]) -> bool:
    ok, _ = append_row(conn, "Sheet1", PROPOSAL_COLUMNS, {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Title": scored["title"],
        "PI": scored["pi"],
        "Budget": scored["budget_raw"],
        "Timeline": scored["timeline_raw"],
        "Total Score": scored["total_score"],
        "Justification": scored.get("justification", ""),
    })
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
#  PDF HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def extract_from_upload(uploaded_file) -> dict[str, Any]:
    """Save an uploaded PDF to a temp file, run extraction_engine."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        metadata = extract_metadata(tmp_path)
        metadata["source_file"] = uploaded_file.name
    finally:
        os.unlink(tmp_path)
    return metadata


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTLY CHARTS
# ═══════════════════════════════════════════════════════════════════════════════


def score_breakdown_chart(scored: dict[str, Any], height: int = 300) -> go.Figure:
    """Horizontal bar chart showing the 4 score components."""
    categories = ["Budget (/30)", "Keywords (/50)", "Timeline (/20)", "PI Bonus"]
    values = [scored["budget_score"], scored["keyword_score"],
              scored["timeline_score"], scored["pi_bonus"]]
    max_vals = [30, 50, 20, 5]
    colors = ["#1e88e5", "#43a047", "#fb8c00", "#8e24aa"]

    fig = go.Figure()
    fig.add_trace(go.Bar(y=categories, x=max_vals, orientation="h",
                         marker_color="rgba(255,255,255,0.07)",
                         name="Max", hoverinfo="skip"))
    fig.add_trace(go.Bar(y=categories, x=values, orientation="h",
                         marker_color=colors, name="Score",
                         text=[f"{v}" for v in values],
                         textposition="inside",
                         textfont=dict(color="white", size=13,
                                       family="Arial Black")))
    fig.update_layout(
        barmode="overlay", height=height,
        margin=dict(l=0, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False, range=[0, 55]),
        yaxis=dict(showgrid=False, autorange="reversed",
                   tickfont=dict(size=12)),
        showlegend=False,
    )
    return fig


def radar_chart(scored: dict[str, Any], height: int = 380) -> go.Figure:
    """
    Five-axis radar chart normalised to 0–100 %.

    Axes:
      1. Budget Efficiency    →  budget_score / 30 × 100
      2. Strategic Alignment  →  keyword_score / 50 × 100
      3. Timeline Realism     →  timeline_score / 20 × 100
      4. PI Experience        →  pi_bonus / 3 × 100  (cap 100)
      5. Overall Compliance   →  total_score / 103 × 100
    """
    axes = ["Budget Efficiency", "Strategic Alignment", "Timeline Realism",
            "PI Experience", "Overall Compliance"]
    values = [
        round(scored["budget_score"] / 30 * 100, 1),
        round(scored["keyword_score"] / 50 * 100, 1),
        round(scored["timeline_score"] / 20 * 100, 1),
        min(round(scored["pi_bonus"] / 3 * 100, 1), 100),
        round(scored["total_score"] / 103 * 100, 1),
    ]
    axes_closed = axes + [axes[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=axes_closed,
        fill="toself", fillcolor="rgba(30,136,229,0.25)",
        line=dict(color="#1e88e5", width=2),
        marker=dict(size=6, color="#42a5f5"),
        name=scored["title"][:30],
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(size=9),
                            gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(tickfont=dict(size=11)),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=height,
        margin=dict(l=60, r=60, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


# ── Proposal detail card ────────────────────────────────────────────────────

def render_proposal_card(scored: dict[str, Any]) -> None:
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
        st.plotly_chart(score_breakdown_chart(scored), width="stretch")

    with st.expander("📊 Detailed Scoring Analytics"):
        st.plotly_chart(radar_chart(scored), width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

gsheets_conn, gsheets_error = _get_gsheets_connection()
_gsheets_online = gsheets_conn is not None and gsheets_error is None

with st.sidebar:
    st.markdown("## ⚒️ Ministry of Coal")
    st.markdown("### R&D Proposal Evaluator")
    st.markdown("---")

    # ── Navigation ───────────────────────────────────────────────────────
    nav = st.radio(
        "Navigation",
        ["🏠 Proposal Evaluator", "📋 View Records", "➕ New Entry"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ── System Status ─────────────────────────────────────────────────────
    st.markdown("### ⚙️ System Status")
    if _gsheets_online:
        st.success("Database: **Connected**", icon="🟢")

        # Professional link buttons (no raw URLs or metadata exposed)
        sheet_url = st.secrets.get("connections", {}).get("gsheets", {}).get("spreadsheet", "")
        drive_folder_id = st.secrets.get("drive", {}).get("folder_id", "")

        if sheet_url:
            st.link_button("📊 Open Evaluation Sheet", sheet_url, use_container_width=True)
        if drive_folder_id:
            drive_url = f"https://drive.google.com/drive/folders/{drive_folder_id}"
            st.link_button("📁 Open Document Vault", drive_url, use_container_width=True)
    else:
        st.error("Database: **Offline**", icon="🔴")
        st.caption("Contact administrator to restore connectivity.")

    st.markdown("---")
    st.caption("Ministry of Coal R&D Evaluator · Built with Streamlit")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA — HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="main-header">
        <h1>R&D Proposal Evaluation Dashboard</h1>
        <p>Ministry of Coal · Automated R&D Scoring & Comparison</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: VIEW RECORDS
# ═══════════════════════════════════════════════════════════════════════════════

if nav == "📋 View Records":
    st.markdown("### 📋 Coal Evaluation Leaderboard")

    if not _gsheets_online:
        st.error(
            "**Google Sheets is offline.** Cannot load records.\n\n"
            + (gsheets_error or "Configure `.streamlit/secrets.toml`."),
            icon="🔴",
        )
        st.stop()

    with st.spinner("Fetching records from Google Sheets…"):
        sheet_df = fetch_sheet_data(gsheets_conn)

    if sheet_df is None or sheet_df.empty:
        st.info("The Google Sheet is empty — no records to display yet.", icon="📭")
    else:
        # Ensure Total Score is numeric for sorting
        if "Total Score" in sheet_df.columns:
            sheet_df["Total Score"] = pd.to_numeric(
                sheet_df["Total Score"], errors="coerce"
            ).fillna(0)
            sheet_df = sheet_df.sort_values("Total Score", ascending=False).reset_index(drop=True)

        # Ensure Innovation is numeric for metrics
        if "Technical Innovation" in sheet_df.columns:
            sheet_df["Technical Innovation"] = pd.to_numeric(
                sheet_df["Technical Innovation"], errors="coerce"
            ).fillna(0)

        # ── Metrics Summary ──────────────────────────────────────────────
        vm1, vm2, vm3, vm4 = st.columns(4)
        with vm1:
            st.metric("📋 Total Proposals", len(sheet_df))
        with vm2:
            if "Technical Innovation" in sheet_df.columns:
                avg_inn = sheet_df["Technical Innovation"].mean()
                st.metric("💡 Avg Innovation", f"{avg_inn:.1f} / 10")
            else:
                st.metric("⛏️ Grades", sheet_df["Coal Grade"].nunique() if "Coal Grade" in sheet_df.columns else "—")
        with vm3:
            if "Total Score" in sheet_df.columns and len(sheet_df) > 0:
                top_score = sheet_df["Total Score"].max()
                st.metric("🏆 Top Score", f"{top_score:.1f} / 100")
            else:
                st.metric("🏆 Top Score", "—")
        with vm4:
            if "Location / Mine" in sheet_df.columns:
                top_org = sheet_df.loc[
                    sheet_df.get("Total Score", pd.Series(dtype=float)).idxmax(),
                    "Location / Mine",
                ] if "Total Score" in sheet_df.columns and len(sheet_df) > 0 else "—"
                st.metric("🏢 Top Organization", str(top_org)[:25])
            else:
                st.metric("🕐 Latest", str(sheet_df["Timestamp"].iloc[-1])[:16] if "Timestamp" in sheet_df.columns else "—")

        st.markdown("")

        # ── Winner callout ────────────────────────────────────────────────
        if "Total Score" in sheet_df.columns and len(sheet_df) > 0:
            winner = sheet_df.iloc[0]
            st.success(
                f"🏆 **#1 Proposal:** {winner.get('Proposal Name', '—')}  "
                f"— Score: **{winner.get('Total Score', 0):.1f} / 100**",
                icon="🏆",
            )

        # ── Search / filter ──────────────────────────────────────────────
        search_term = st.text_input(
            "🔍 Search records",
            placeholder="Type to filter across all columns…",
            key="view_search",
        )
        display_df = sheet_df.copy()
        if search_term:
            mask = sheet_df.astype(str).apply(
                lambda col: col.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            display_df = sheet_df[mask]
            st.caption(f"Showing {len(display_df)} of {len(sheet_df)} records.")

        # Add rank column
        display_df = display_df.copy()
        display_df.insert(0, "Rank", range(1, len(display_df) + 1))

        # ── PyArrow fix: coerce ALL columns to str before display ────────
        display_df = display_df.astype(str)

        # ── Styled dataframe with gradient on Total Score & Innovation ───
        # Re-cast numeric columns back so gradients work on numbers
        for _nc in ["Total Score", "Technical Innovation"]:
            if _nc in display_df.columns:
                display_df[_nc] = pd.to_numeric(display_df[_nc], errors="coerce").fillna(0)

        styled = display_df.style
        if "Total Score" in display_df.columns:
            styled = styled.background_gradient(
                subset=["Total Score"],
                cmap="RdYlGn",
                vmin=0,
                vmax=100,
            )
        if "Technical Innovation" in display_df.columns:
            styled = styled.background_gradient(
                subset=["Technical Innovation"],
                cmap="YlOrRd",
                vmin=1,
                vmax=10,
            )

        st.dataframe(
            styled,
            width="stretch",
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn(width="small"),
                "Total Score": st.column_config.NumberColumn(format="%.1f"),
                "PDF Link": st.column_config.LinkColumn(display_text="📄 View"),
            },
        )

        # ── CSV download ─────────────────────────────────────────────────
        st.download_button(
            "📥 Export Leaderboard as CSV",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="coal_eval_leaderboard.csv",
            mime="text/csv",
        )

    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: NEW ENTRY (Coal Evaluation Form)
# ═══════════════════════════════════════════════════════════════════════════════

if nav == "➕ New Entry":
    st.markdown("### ➕ New Coal Evaluation Entry")

    if not _gsheets_online:
        st.error(
            "**Google Sheets is offline.** Cannot submit entries.\n\n"
            + (gsheets_error or "Configure `.streamlit/secrets.toml`."),
            icon="🔴",
        )
        st.stop()

    _drive_folder_id = st.secrets.get("drive", {}).get("folder_id", "")

    # ═══ SMART EXTRACTION & AI ANALYSIS ═════════════════════════════════════════
    with st.expander("🧠 Smart Extract & AI Scoring", expanded=True):
        st.caption(
            "Upload a proposal PDF. The engine will extract key fields "
            "**and** the Gemini AI will score the proposal automatically."
        )
        scan_pdf = st.file_uploader(
            "Drop a Proposal PDF to scan",
            type=["pdf"],
            key="smart_extract_pdf",
            help="Extracts Title, PI, Organization, Coal Grade — then sends text to Gemini for scoring.",
        )

        if scan_pdf is not None:
            btn_col1, btn_col2 = st.columns(2)
            run_extract = btn_col1.button("🔍 Extract Fields", use_container_width=True)
            run_ai = btn_col2.button("🧠 AI-Score Proposal", type="primary", use_container_width=True)

            # ── Shared: write PDF to temp file for extraction engine ─────
            _tmp_path = None
            if run_extract or run_ai:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(scan_pdf.getbuffer())
                    _tmp_path = tmp.name

            # ── Field Extraction ─────────────────────────────────────────
            if run_extract and _tmp_path:
                with st.spinner("Extracting fields from PDF…"):
                    try:
                        result = smart_extract_for_form(_tmp_path)
                    finally:
                        os.unlink(_tmp_path)
                        _tmp_path = None

                st.session_state["_smart_fill"] = result
                st.session_state["_smart_pdf_name"] = scan_pdf.name

                # Store extracted text for downstream semantic analysis
                from extraction_engine import extract_text as _ext_text
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as _txt_tmp:
                    _txt_tmp.write(scan_pdf.getbuffer())
                    _txt_tmp_path = _txt_tmp.name
                try:
                    _raw, _ = _ext_text(_txt_tmp_path)
                    st.session_state["extracted_text"] = _raw
                finally:
                    if os.path.exists(_txt_tmp_path):
                        os.unlink(_txt_tmp_path)

                flags = result["confidence_flags"]
                conf = result["confidence_score"]
                method = result["extraction_method"]

                st.success(
                    f"**Extraction complete** — {conf} fields detected  ·  Method: {method}",
                    icon="✅",
                )
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown("**Detected Fields:**")
                    for field, found in flags.items():
                        ic = "✅" if found else "❌"
                        st.markdown(f"{ic}  {field.replace('_', ' ').title()}")
                with rc2:
                    st.markdown("**Extracted Values:**")
                    st.markdown(f"**Title:** {result['proposal_name'] or '—'}")
                    st.markdown(f"**PI:** {result['evaluator'] or '—'}")
                    st.markdown(f"**Organization:** {result['raw_organization']}")
                    st.markdown(f"**Coal Grade:** {result['raw_coal_grade']} → {result['coal_grade']}")
                    st.markdown(f"**Budget:** {result['raw_budget']}")
                    st.markdown(f"**Timeline:** {result['raw_timeline']}")

            # ── AI Semantic Scoring (existing 4-axis scorer) ─────────────
            if run_ai and _tmp_path:
                gemini_key = _resolve_gemini_key()
                if not gemini_key:
                    st.error(
                        "**Gemini API key not configured.**\n\n"
                        "Add this to `.streamlit/secrets.toml`:\n\n"
                        '```toml\nGEMINI_API_KEY = "your-key-here"\n```',
                        icon="🔑",
                    )
                elif not semantic_scorer.is_available():
                    st.error(
                        "`google-generativeai` is not installed.\n\n"
                        "Run: `pip install google-generativeai`",
                        icon="❌",
                    )
                else:
                    with st.spinner("🧠 Gemini AI is evaluating the proposal…"):
                        try:
                            form_data = smart_extract_for_form(_tmp_path)
                            st.session_state["_smart_fill"] = form_data
                            st.session_state["_smart_pdf_name"] = scan_pdf.name

                            from extraction_engine import extract_text
                            raw_text, _ = extract_text(_tmp_path)

                            ai_result = semantic_scorer.analyze_proposal(
                                raw_text, gemini_key,
                            )
                            st.session_state["_ai_scores"] = ai_result
                            st.session_state["extracted_text"] = raw_text
                        except Exception as exc:
                            st.error(f"AI analysis failed: {exc}", icon="❌")
                            ai_result = None
                        finally:
                            os.unlink(_tmp_path)
                            _tmp_path = None

                    if ai_result:
                        st.success(
                            f"**AI Scoring complete** · Model: `{ai_result['model']}`",
                            icon="🧠",
                        )

                        ai1, ai2, ai3, ai4 = st.columns(4)
                        ai1.metric("💡 Innovation", f"{ai_result['technical_innovation']} / 10")
                        ai2.metric("💰 Economic", f"{ai_result['economic_viability']} / 10")
                        ai3.metric("🌱 Environmental", f"{ai_result['environmental_sustainability']} / 10")
                        ai4.metric("🏛️ Ministry", f"{ai_result['ministry_alignment']} / 10")

                        st.markdown("---")
                        st.markdown("**📝 AI Reasoning:**")
                        st.info(ai_result["reasoning"], icon="🧠")

            # Clean up temp file if still around
            if _tmp_path and os.path.exists(_tmp_path):
                os.unlink(_tmp_path)

        if st.session_state.get("_smart_fill") or st.session_state.get("_ai_scores"):
            st.info("⬇️ Fields and/or scores have been pre-filled below. Review and submit.", icon="⬇️")

    # ═══ 🧠 AI TECHNICAL AUDIT — SEMANTIC ANALYSIS ═══════════════════════════
    with st.expander("🧠 AI Technical Audit", expanded=False):
        st.caption(
            "Run a **Semantic Analysis** on the uploaded PDF.  The Gemini AI "
            "will act as a **Ministry of Coal Technical Auditor** and return "
            "Innovation, Feasibility, and Impact scores (1–10) plus a "
            "2-sentence technical summary.  "
            "Scores are auto-filled into the evaluation sliders below."
        )

        # The audit needs a PDF — reuse the same uploader from Smart Extract
        _audit_pdf = st.session_state.get("smart_extract_pdf")
        if _audit_pdf is None:
            st.info(
                "⬆️ Upload a PDF in the **Smart Extract & AI Scoring** panel "
                "above first, then click the button below.",
                icon="📄",
            )

        run_semantic = st.button(
            "🔍 Run AI Semantic Audit",
            type="primary",
            use_container_width=True,
            disabled=(_audit_pdf is None),
            key="run_semantic_btn",
        )

        if run_semantic and _audit_pdf is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(_audit_pdf.getbuffer())
                _sem_path = tmp.name
            try:
                with st.spinner("🧠 Gemini 1.5 Pro is auditing the proposal…"):
                    from extraction_engine import extract_text
                    raw_text, _ = extract_text(_sem_path)
                    sem_result = get_semantic_analysis(raw_text)
                    st.session_state["_semantic_result"] = sem_result
            except (RuntimeError, ValueError) as exc:
                # Config / validation errors (missing API key, short text, etc.)
                st.error(
                    f"**Semantic Audit configuration error:** {exc}",
                    icon="🔑",
                )
            except Exception as exc:
                # Gemini API or network errors — show a user-friendly message
                err_msg = str(exc)
                if "404" in err_msg or "not found" in err_msg.lower():
                    st.error(
                        "**Gemini API model not found (404).** "
                        "The requested model may have been deprecated.\n\n"
                        f"Details: `{err_msg}`",
                        icon="❌",
                    )
                elif "403" in err_msg or "permission" in err_msg.lower():
                    st.error(
                        "**Gemini API permission denied (403).** "
                        "Check that your API key is valid and has access.\n\n"
                        f"Details: `{err_msg}`",
                        icon="🔒",
                    )
                elif "429" in err_msg or "quota" in err_msg.lower():
                    st.error(
                        "**Gemini API rate limit / quota exceeded.** "
                        "Wait a moment and try again, or upgrade your API plan.\n\n"
                        f"Details: `{err_msg}`",
                        icon="⏳",
                    )
                else:
                    st.error(
                        f"**AI Semantic Audit failed:** {err_msg}",
                        icon="❌",
                    )
            finally:
                if os.path.exists(_sem_path):
                    os.unlink(_sem_path)

        # ── Display results & auto-fill ──────────────────────────────────
        sem = st.session_state.get("_semantic_result")
        if sem:
            # Map semantic scores → evaluation sliders
            st.session_state.setdefault("_ai_scores", {})
            st.session_state["_ai_scores"]["technical_innovation"] = sem["innovation_score"]
            st.session_state["_ai_scores"]["economic_viability"] = sem["feasibility_score"]
            st.session_state["_ai_scores"]["environmental_sustainability"] = sem["impact_score"]

            # ── Confidence Card ──────────────────────────────────────
            inn = sem["innovation_score"]
            feas = sem["feasibility_score"]
            imp = sem["impact_score"]
            avg = round((inn + feas + imp) / 3, 1)

            # Colour coding
            if avg >= 7:
                card_border = "#43a047"
                card_bg = "rgba(67,160,71,0.08)"
                verdict = "🟢 High Confidence"
            elif avg >= 4:
                card_border = "#fb8c00"
                card_bg = "rgba(251,140,0,0.08)"
                verdict = "🟡 Moderate Confidence"
            else:
                card_border = "#e53935"
                card_bg = "rgba(229,57,53,0.08)"
                verdict = "🔴 Low Confidence"

            st.markdown(
                f"""
                <div style="
                    background: {card_bg};
                    border-left: 5px solid {card_border};
                    border-radius: 10px;
                    padding: 1.2rem 1.5rem;
                    margin: 1rem 0;
                ">
                    <h4 style="margin:0 0 .6rem 0; color:#fff;">
                        🧠 Semantic Audit Card &nbsp;·&nbsp; {verdict}
                    </h4>
                    <table style="width:100%; color:#e8eaf6; font-size:.95rem;">
                        <tr>
                            <td><strong>💡 Innovation Score</strong></td>
                            <td style="text-align:right; font-weight:700; font-size:1.2rem;">
                                {inn} / 10
                            </td>
                        </tr>
                        <tr>
                            <td><strong>⚙️ Feasibility Score</strong></td>
                            <td style="text-align:right; font-weight:700; font-size:1.2rem;">
                                {feas} / 10
                            </td>
                        </tr>
                        <tr>
                            <td><strong>🌍 Impact Score</strong></td>
                            <td style="text-align:right; font-weight:700; font-size:1.2rem;">
                                {imp} / 10
                            </td>
                        </tr>
                    </table>
                    <hr style="border-color:rgba(255,255,255,0.1); margin:.8rem 0;">
                    <p style="margin:0; color:#cfd8dc; font-size:.92rem;">
                        📝 <em>{sem['technical_summary']}</em>
                    </p>
                    <p style="margin:.5rem 0 0 0; color:#90a4ae; font-size:.8rem;">
                        Model: <code>{sem['model']}</code>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.success(
                "✨ Innovation, Feasibility, and Impact scores have been "
                "auto-filled into the evaluation sliders below.",
                icon="⬇️",
            )

    # ═══ 🤖 DEEP AI ANALYSIS — RUN SEMANTIC ANALYSIS ═════════════════════════
    st.divider()
    st.subheader("🤖 Deep AI Analysis")

    run_deep_analysis = st.button(
        "🔍 Run Semantic Analysis",
        type="primary",
        use_container_width=True,
        key="run_deep_semantic_btn",
    )

    if run_deep_analysis:
        if not st.session_state.get("extracted_text"):
            st.warning(
                "⬆️ Please upload and extract a PDF first using the "
                "**Smart Extract & AI Scoring** panel above.",
                icon="⚠️",
            )
        else:
            with st.spinner("Ministry AI is analyzing the proposal…"):
                try:
                    deep_result = get_semantic_analysis(
                        st.session_state["extracted_text"]
                    )
                    st.session_state["_deep_analysis"] = deep_result
                    st.success(
                        f"**Semantic Analysis complete!** · Model: `{deep_result['model']}`",
                        icon="✅",
                    )
                    st.write(deep_result)
                except Exception as exc:
                    st.error(
                        f"**Semantic Analysis failed:** {exc}",
                        icon="❌",
                    )

    # Show persisted results on reruns
    if st.session_state.get("_deep_analysis") and not run_deep_analysis:
        deep = st.session_state["_deep_analysis"]
        st.success(
            f"**Semantic Analysis results** · Model: `{deep['model']}`",
            icon="✅",
        )
        st.write(deep)

    # ── Read pre-fill values from session state (or defaults) ──────────
    sf = st.session_state.get("_smart_fill", {})
    ai = st.session_state.get("_ai_scores", {})
    _pf_name = sf.get("proposal_name", "")
    _pf_grade = sf.get("coal_grade", "D")
    _pf_location = sf.get("location", "")
    _pf_evaluator = sf.get("evaluator", "")

    # AI-suggested slider defaults (fall back to 5)
    _pf_innovation = ai.get("technical_innovation", 5)
    _pf_economic = ai.get("economic_viability", 5)
    _pf_environmental = ai.get("environmental_sustainability", 5)
    _pf_alignment = ai.get("ministry_alignment", 5)

    GRADE_OPTIONS = ["A", "B", "C", "D", "E", "F", "G",
                     "Semi-Coking", "Coking", "Washery"]
    _pf_grade_idx = GRADE_OPTIONS.index(_pf_grade) if _pf_grade in GRADE_OPTIONS else 3

    st.markdown("---")
    st.caption(
        "Fill in the coal sample evaluation form below. Data is appended "
        "to the connected Google Sheet. If a PDF is attached, it is "
        "uploaded to Google Drive and the link is saved with the record."
    )

    with st.form("coal_entry_form", clear_on_submit=True):
        # ── Proposal & PDF ───────────────────────────────────────────
        st.markdown("##### Proposal")
        proposal_name = st.text_input(
            "Proposal Name",
            value=_pf_name,
            placeholder="e.g. Underground Methane Capture — Phase II",
        )
        pdf_file = st.file_uploader(
            "Attach Proposal PDF (optional)",
            type=["pdf"],
            key="entry_pdf_upload",
            help="The PDF will be uploaded to Google Drive and linked in the record.",
        )

        # ── Sample Details ───────────────────────────────────────────
        st.markdown("##### Sample Details")
        fc1, fc2 = st.columns(2)
        with fc1:
            coal_grade = st.selectbox(
                "Coal Grade",
                GRADE_OPTIONS,
                index=_pf_grade_idx,
                help="Indian coal grade classification.",
            )
            ash = st.number_input("Ash %", 0.0, 100.0, 15.0, step=0.1)
            moisture = st.number_input("Moisture %", 0.0, 100.0, 8.0, step=0.1)
        with fc2:
            volatile = st.number_input("Volatile Matter %", 0.0, 100.0, 25.0, step=0.1)
            fixed_carbon = st.number_input("Fixed Carbon %", 0.0, 100.0, 45.0, step=0.1)
            gcv = st.number_input("GCV (kcal/kg)", 0, 10000, 5500, step=50)

        # ── Metadata ─────────────────────────────────────────────────
        st.markdown("##### Metadata")
        fm1, fm2 = st.columns(2)
        with fm1:
            location = st.text_input(
                "Location / Mine",
                value=_pf_location,
                placeholder="e.g. Jharia Coalfield",
            )
        with fm2:
            evaluator = st.text_input(
                "Evaluator Name",
                value=_pf_evaluator,
                placeholder="e.g. Dr. Sharma",
            )

        # ── Evaluation Scoring ───────────────────────────────────────
        st.markdown("##### 📊 Evaluation Scoring")
        if st.session_state.get("_ai_scores"):
            st.caption(
                "✨ Sliders pre-filled by **Gemini AI**. "
                "Adjust if needed before submitting. "
                "Weights: Innovation 30% · Economic 25% · Environmental 25% · Ministry 20%"
            )
        else:
            st.caption(
                "Rate each dimension on a 1–10 scale. "
                "Weights: Innovation 30% · Economic 25% · Environmental 25% · Ministry 20%"
            )
        es1, es2 = st.columns(2)
        with es1:
            sc_innovation = st.slider(
                "💡 Technical Innovation", 1, 10, _pf_innovation,
                help="Novelty of approach, use of emerging tech (AI/IoT/sensors).",
            )
            sc_economic = st.slider(
                "💰 Economic Viability", 1, 10, _pf_economic,
                help="Cost effectiveness, ROI potential, budget realism.",
            )
        with es2:
            sc_environmental = st.slider(
                "🌱 Environmental Sustainability", 1, 10, _pf_environmental,
                help="Carbon reduction, waste management, ecological impact.",
            )
            sc_alignment = st.slider(
                "🏛️ Ministry Alignment", 1, 10, _pf_alignment,
                help="Alignment with Coal Ministry 2026 strategic goals.",
            )

        # ── Live score preview ────────────────────────────────────────
        _live_score = calculate_eval_score(
            sc_innovation, sc_economic, sc_environmental, sc_alignment,
        )
        _score_colour = "🟢" if _live_score >= 70 else "🟡" if _live_score >= 40 else "🔴"
        st.markdown(
            f"**{_score_colour} Weighted Total Score: "
            f"`{_live_score}` / 100**"
        )

        submitted = st.form_submit_button("📤 Submit Entry", type="primary")

    if submitted:
        pdf_link = "—"

        # ── Step 1: Upload PDF to Drive (if provided + folder configured)
        if pdf_file is not None:
            if not _drive_folder_id:
                st.warning(
                    "PDF attached but `drive.folder_id` is not set in "
                    "`secrets.toml`. Skipping Drive upload.",
                    icon="⚠️",
                )
            elif not _DRIVE_LIB_AVAILABLE:
                st.warning(
                    "PDF attached but `google-api-python-client` is not "
                    "installed. Skipping Drive upload.",
                    icon="⚠️",
                )
            else:
                with st.spinner("Uploading PDF to Google Drive…"):
                    link, drive_msg = upload_pdf_to_drive(
                        pdf_file.getvalue(),
                        pdf_file.name,
                        _drive_folder_id,
                    )
                if link:
                    pdf_link = link
                    st.success(f"📄 PDF uploaded: [View on Drive]({link})", icon="✅")
                else:
                    st.error(drive_msg, icon="❌")

        # ── Step 2: Calculate final score & append row ────────────────
        total_score = calculate_eval_score(
            sc_innovation, sc_economic, sc_environmental, sc_alignment,
        )
        row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Proposal Name": proposal_name or "—",
            "Coal Grade": coal_grade,
            "Ash %": ash,
            "Moisture %": moisture,
            "Volatile Matter %": volatile,
            "Fixed Carbon %": fixed_carbon,
            "GCV (kcal/kg)": gcv,
            "Location / Mine": location or "—",
            "Evaluator": evaluator or "—",
            "Technical Innovation": sc_innovation,
            "Economic Viability": sc_economic,
            "Environmental Sustainability": sc_environmental,
            "Ministry Alignment": sc_alignment,
            "Total Score": total_score,
            "PDF Link": pdf_link,
        }
        with st.spinner("Submitting to Google Sheets…"):
            ok, msg = append_row(gsheets_conn, "Sheet1", EVAL_COLUMNS, row)
        if ok:
            # Clear smart-fill state after successful submit
            st.session_state.pop("_smart_fill", None)
            st.session_state.pop("_smart_pdf_name", None)
            st.success("✅ Entry submitted and cached data refreshed!", icon="✅")
            st.balloons()
        else:
            st.error(msg, icon="❌")

    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: PROPOSAL EVALUATOR  (default — requires PDF uploads)
# ═══════════════════════════════════════════════════════════════════════════════

# File uploader lives in the main area when on the evaluator page
if not st.session_state.get("_files_from_sidebar"):
    # Show uploader in sidebar check
    pass

# We need the files — check the sidebar uploader
uploaded_files = st.sidebar.file_uploader(
    "Upload R&D Proposal PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload two or more PDF proposals to compare them.",
    key="pdf_uploader_main",
)

if not uploaded_files:
    st.info(
        "👈  **Upload two or more PDF proposals** from the sidebar to get started.",
        icon="📄",
    )
    st.stop()

if "scored_proposals" not in st.session_state:
    st.session_state.scored_proposals = []
    st.session_state.processed_names = set()

new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_names]

if new_files:
    for uf in new_files:
        try:
            with st.status(f"Processing **{uf.name}** …", expanded=True) as status:
                st.write("Extracting digital text (PyMuPDF)…")
                metadata = extract_from_upload(uf)

                # Also store the raw text for semantic analysis
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as _ev_tmp:
                    _ev_tmp.write(uf.getbuffer())
                    _ev_tmp_path = _ev_tmp.name
                try:
                    from extraction_engine import extract_text as _ev_ext
                    _ev_raw, _ = _ev_ext(_ev_tmp_path)
                    st.session_state["extracted_text"] = _ev_raw
                finally:
                    if os.path.exists(_ev_tmp_path):
                        os.unlink(_ev_tmp_path)

                st.write(f"Extraction method: **{metadata.get('extraction_method', 'Digital Text Layer')}**")
                st.write("Calculating score…")

                scored = calculate_score(metadata)
                scored["keywords_all"] = metadata.get("keywords", [])
                scored["source_file"] = metadata.get("source_file", uf.name)
                scored["extraction_method"] = metadata.get("extraction_method", "Digital Text Layer")

                st.session_state.scored_proposals.append(scored)
                st.session_state.processed_names.add(uf.name)
                status.update(label=f"**{uf.name}** — done!", state="complete")

        except Exception as exc:
            st.error(f"Failed to process **{uf.name}**: {exc}", icon="❌")

# Handle file removals
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

proposals_ranked = sorted(proposals, key=lambda s: s["total_score"], reverse=True)
title_map: dict[str, dict] = {s["title"]: s for s in proposals_ranked}
titles = list(title_map.keys())


# ═══════════════════════════════════════════════════════════════════════════════
#  🤖 DEEP AI ANALYSIS — PROPOSAL EVALUATOR PAGE
# ═══════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("🤖 Deep AI Analysis")

run_eval_semantic = st.button(
    "🔍 Run Semantic Analysis",
    type="primary",
    use_container_width=True,
    key="run_eval_semantic_btn",
)

if run_eval_semantic:
    if not st.session_state.get("extracted_text"):
        st.warning(
            "👈 Upload a PDF proposal from the sidebar first. "
            "The text will be extracted automatically.",
            icon="⚠️",
        )
    else:
        with st.spinner("Ministry AI is analyzing the proposal…"):
            try:
                deep_result = get_semantic_analysis(
                    st.session_state["extracted_text"]
                )
                st.session_state["_deep_analysis_eval"] = deep_result
                st.success(
                    f"**Semantic Analysis complete!** · Model: `{deep_result['model']}`",
                    icon="✅",
                )
                st.write(deep_result)
            except Exception as exc:
                st.error(
                    f"**Semantic Analysis failed:** {exc}",
                    icon="❌",
                )

# Show persisted results on reruns
if st.session_state.get("_deep_analysis_eval") and not run_eval_semantic:
    deep_ev = st.session_state["_deep_analysis_eval"]
    st.success(
        f"**Semantic Analysis results** · Model: `{deep_ev['model']}`",
        icon="✅",
    )
    st.write(deep_ev)


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATOR TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab_dash, tab_analytics, tab_compare = st.tabs([
    "📊  Dashboard",
    "🔬  Analytics Vault",
    "⚖️  Compare Mode",
])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_dash:

    best_proposal = proposals_ranked[0]
    st.success(
        f"🏆 **#1 Ranked Proposal:** {best_proposal['title']}  "
        f"— Score: **{best_proposal['total_score']} pts**",
        icon="🏆",
    )
    st.markdown("")

    # ── Metric cards ─────────────────────────────────────────────────────
    best = proposals_ranked[0]
    budgets = [(s["title"], s["budget_value"]) for s in proposals_ranked if s["budget_value"] > 0]
    best_budget_title = min(budgets, key=lambda x: x[1])[0] if budgets else "N/A"
    best_budget_val = min(budgets, key=lambda x: x[1])[1] if budgets else 0

    best_timeline = min(
        (s for s in proposals_ranked if s["timeline_months"] > 0),
        key=lambda s: s["timeline_months"],
        default=None,
    )

    m0, m1, m2, m3 = st.columns(4)
    with m0:
        st.metric(label="📋 Total Proposals",
                  value=str(len(proposals_ranked)),
                  delta="Processed")
    with m1:
        st.metric(label="🏆 Top Score",
                  value=f"{best['total_score']} pts",
                  delta=best["title"][:40])
    with m2:
        fmt_budget = f"₹{best_budget_val:,.0f}" if best_budget_val else "N/A"
        st.metric(label="💰 Best Budget (Lowest)",
                  value=fmt_budget,
                  delta=best_budget_title[:40])
    with m3:
        tl_val = f"{best_timeline['timeline_months']} months" if best_timeline else "N/A"
        tl_name = best_timeline["title"][:40] if best_timeline else ""
        st.metric(label="⏱️ Shortest Timeline", value=tl_val, delta=tl_name)

    st.markdown("")

    # ── Ranked table ─────────────────────────────────────────────────────
    st.markdown("### Ranked Proposals")
    df = pd.DataFrame([
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

    st.dataframe(
        df.astype(str), width="stretch", hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Total Score": st.column_config.NumberColumn(format="%.1f"),
            "Justification": st.column_config.TextColumn(width="large"),
        },
    )

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Leaderboard as CSV",
        data=csv_data,
        file_name="proposal_leaderboard.csv",
        mime="text/csv",
    )

    # ── Submit to Google Sheets ──────────────────────────────────────────
    st.markdown("---")
    if _gsheets_online:
        st.markdown("### 🗄️ Submit to Ministry Database")
        submit_choice = st.selectbox(
            "Select proposal to submit",
            titles,
            key="submit_select",
        )
        if st.button("📤 Submit to Ministry Database", type="primary"):
            with st.spinner("Writing to Google Sheets…"):
                success = submit_to_sheets(gsheets_conn, title_map[submit_choice])
            if success:
                st.success(
                    f"**{submit_choice}** submitted to the Ministry database!",
                    icon="✅",
                )
            else:
                st.error(
                    "Failed to write to Google Sheets.\n\n"
                    f"Ensure `{SERVICE_ACCOUNT_EMAIL}` has **Editor** access.",
                    icon="❌",
                )
    else:
        st.info(
            "**Database persistence is offline.** Configure Google Sheets "
            "credentials in `.streamlit/secrets.toml` to enable the "
            "'Submit to Ministry Database' feature.",
            icon="🗄️",
        )

    # ── Detailed view ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Detailed Proposal View")
    selected = st.selectbox("Select a proposal to inspect", titles, key="detail_select")
    if selected:
        render_proposal_card(title_map[selected])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — ANALYTICS VAULT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown("### 🔬 Scoring Analytics — Radar Charts")
    st.caption(
        "Each radar chart visualises five normalised metrics (0–100 %): "
        "Budget Efficiency, Strategic Alignment, Timeline Realism, "
        "PI Experience, and Overall Compliance."
    )
    st.markdown("")

    cols = st.columns(2)
    for idx, scored in enumerate(proposals_ranked):
        with cols[idx % 2]:
            st.markdown(f"**{scored['title'][:50]}**")
            st.markdown(
                f'<div class="score-badge">{scored["total_score"]} pts</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                radar_chart(scored, height=350),
                width="stretch",
                key=f"radar_{idx}",
            )
            st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — COMPARE MODE
# ═══════════════════════════════════════════════════════════════════════════════
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

            cmp_df = pd.DataFrame({
                "Metric": [
                    "PI", "PI Rank", "Budget", "Timeline",
                    "Priority Keywords Matched",
                    "Budget Score (/30)", "Keyword Score (/50)",
                    "Timeline Score (/20)", "PI Bonus", "TOTAL SCORE",
                ],
                pick_a: [
                    a["pi"], a["pi_rank"], a["budget_raw"], a["timeline_raw"],
                    ", ".join(a["matched_keywords"]) or "None",
                    a["budget_score"], a["keyword_score"],
                    a["timeline_score"], f"+{a['pi_bonus']}", a["total_score"],
                ],
                pick_b: [
                    b["pi"], b["pi_rank"], b["budget_raw"], b["timeline_raw"],
                    ", ".join(b["matched_keywords"]) or "None",
                    b["budget_score"], b["keyword_score"],
                    b["timeline_score"], f"+{b['pi_bonus']}", b["total_score"],
                ],
            })
            st.dataframe(cmp_df.astype(str), width="stretch", hide_index=True)

            st.markdown("#### Score Breakdown")
            ch1, ch2 = st.columns(2)
            with ch1:
                st.markdown(f"**{pick_a}**")
                st.plotly_chart(score_breakdown_chart(a, height=250),
                               width="stretch", key="cmp_chart_a")
            with ch2:
                st.markdown(f"**{pick_b}**")
                st.plotly_chart(score_breakdown_chart(b, height=250),
                               width="stretch", key="cmp_chart_b")

            # ── Radar overlay comparison ─────────────────────────────────
            st.markdown("#### Radar Comparison")
            overlay_fig = go.Figure()
            for s, color, fill in [(a, "#1e88e5", "rgba(30,136,229,0.2)"),
                                    (b, "#e91e63", "rgba(233,30,99,0.2)")]:
                axes = ["Budget Efficiency", "Strategic Alignment",
                         "Timeline Realism", "PI Experience", "Overall Compliance"]
                vals = [
                    round(s["budget_score"] / 30 * 100, 1),
                    round(s["keyword_score"] / 50 * 100, 1),
                    round(s["timeline_score"] / 20 * 100, 1),
                    min(round(s["pi_bonus"] / 3 * 100, 1), 100),
                    round(s["total_score"] / 103 * 100, 1),
                ]
                overlay_fig.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=axes + [axes[0]],
                    fill="toself", fillcolor=fill,
                    line=dict(color=color, width=2),
                    name=s["title"][:30],
                ))
            overlay_fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100],
                                    gridcolor="rgba(255,255,255,0.1)"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                height=420,
                margin=dict(l=60, r=60, t=40, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(x=0.5, y=-0.15, xanchor="center",
                            orientation="h"),
            )
            st.plotly_chart(overlay_fig, width="stretch",
                           key="cmp_radar_overlay")

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
