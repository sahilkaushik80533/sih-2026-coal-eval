#!/usr/bin/env python3
"""
generate_sample.py
==================
Creates a structured, multi-page research-proposal PDF (sample_proposal.pdf)
designed to exercise every extraction rule in extraction_engine.py.

Pages
-----
1  Cover page   – Title, PI name, affiliation, submission date
2  Summary page – Abstract with domain-specific technical keywords
3  Budget page  – Financial table with ₹ / Rs values and project timeline

Dependencies: fpdf2 (pip install fpdf2)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fpdf import FPDF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _system_font_dir() -> Path:
    """Return the Windows Fonts directory."""
    windir = os.environ.get("WINDIR", r"C:\Windows")
    return Path(windir) / "Fonts"


class ProposalPDF(FPDF):
    """Light wrapper around FPDF with a Unicode-capable system font loaded."""

    FONT_FAMILY = "UniFont"

    def __init__(self) -> None:
        super().__init__()
        fonts_dir = _system_font_dir()

        # Prefer Arial Unicode MS (full ₹ support); fall back to regular Arial
        candidates = [
            ("ARIALUNI.ttf", None),      # Regular only (no bold variant)
            ("arial.ttf", "arialbd.ttf"), # Regular + Bold
        ]
        loaded = False
        for regular, bold in candidates:
            reg_path = fonts_dir / regular
            if reg_path.exists():
                self.add_font(self.FONT_FAMILY, "", str(reg_path))
                # Use the same file for bold if no dedicated bold variant
                bold_path = fonts_dir / bold if bold else reg_path
                if bold_path.exists():
                    self.add_font(self.FONT_FAMILY, "B", str(bold_path))
                else:
                    self.add_font(self.FONT_FAMILY, "B", str(reg_path))
                # Italic – reuse regular
                self.add_font(self.FONT_FAMILY, "I", str(reg_path))
                loaded = True
                break

        if not loaded:
            raise RuntimeError(
                "No suitable Unicode TTF font found in system Fonts directory. "
                "Please install Arial Unicode MS or place a .ttf font manually."
            )

    # --- convenience writers ------------------------------------------------

    def section_title(self, text: str, size: int = 14) -> None:
        self.set_font("UniFont", "B", size)
        self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text: str, size: int = 11) -> None:
        self.set_font("UniFont", "", size)
        self.multi_cell(0, 7, text)
        self.ln(2)

    def label_value(self, label: str, value: str, size: int = 11) -> None:
        self.set_font("UniFont", "B", size)
        self.cell(55, 8, f"{label}:")
        self.set_font("UniFont", "", size)
        self.cell(0, 8, value, new_x="LMARGIN", new_y="NEXT")

    def add_table_row(self, cells: list[str], bold: bool = False) -> None:
        style = "B" if bold else ""
        self.set_font("UniFont", style, 10)
        col_w = (self.w - self.l_margin - self.r_margin) / len(cells)
        for cell_text in cells:
            self.cell(col_w, 8, cell_text, border=1)
        self.ln()


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def _page_cover(pdf: ProposalPDF) -> None:
    """Page 1 – Cover / title page."""
    pdf.add_page()
    pdf.ln(25)

    # Title
    pdf.set_font("UniFont", "B", 20)
    pdf.multi_cell(
        0, 12,
        "Advanced Methane Detection and Safety Automation\n"
        "System for Underground Coal Mining Operations",
        align="C",
    )
    pdf.ln(15)

    # Metadata block
    pdf.label_value("Principal Investigator", "Dr. Ananya Sharma")
    pdf.label_value("Co-PI", "Prof. Rajesh Kumar")
    pdf.label_value("Affiliation", "Indian Institute of Technology, Dhanbad")
    pdf.label_value("Submitted by", "Dr. Ananya Sharma, IIT Dhanbad")
    pdf.label_value("Date", "15 March 2026")
    pdf.label_value("Proposal ID", "COAL/RD/2026/0042")


def _page_summary(pdf: ProposalPDF) -> None:
    """Page 2 – Project summary & technical keywords."""
    pdf.add_page()
    pdf.section_title("1. Project Summary")
    pdf.body_text(
        "This research proposal outlines an integrated approach to real-time "
        "Methane detection and Safety automation in underground coal mines. "
        "The project leverages IoT sensor networks, edge computing, and "
        "machine-learning-based anomaly detection to provide early warnings "
        "of hazardous gas concentrations during Excavation operations."
    )

    pdf.section_title("2. Technical Objectives")
    pdf.body_text(
        "• Design a low-power wireless sensor mesh for continuous Methane monitoring.\n"
        "• Develop Automation firmware for emergency ventilation control.\n"
        "• Integrate predictive analytics for proactive Safety management.\n"
        "• Validate the system through controlled Underground Mining field trials.\n"
        "• Explore Clean Coal processing techniques to reduce emissions.\n"
        "• Implement advanced Excavation monitoring with LiDAR mapping."
    )

    pdf.section_title("3. Expected Outcomes")
    pdf.body_text(
        "The project aims to reduce mine-accident fatalities by at least 40 % "
        "through early detection and automated response. A deployable prototype "
        "will be delivered within the project timeline, along with open-source "
        "software tools for Safety analytics."
    )


def _page_budget(pdf: ProposalPDF) -> None:
    """Page 3 – Budget breakdown and timeline."""
    pdf.add_page()
    pdf.section_title("4. Budget Estimate")

    # Budget table
    pdf.add_table_row(["Item", "Description", "Amount"], bold=True)
    pdf.add_table_row(["Equipment", "IoT Sensors & Edge Devices", "₹25,00,000"])
    pdf.add_table_row(["Personnel", "Research Staff (3 years)", "₹45,00,000"])
    pdf.add_table_row(["Travel", "Field Trials & Conferences", "₹8,00,000"])
    pdf.add_table_row(["Consumables", "Lab Materials & Supplies", "₹7,00,000"])
    pdf.add_table_row(["Contingency", "Miscellaneous Expenses", "₹5,00,000"])
    pdf.ln(4)

    pdf.set_font("UniFont", "B", 12)
    pdf.cell(0, 10, "Total Budget: ₹90,00,000 (Rs 90,00,000)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.section_title("5. Project Timeline")
    pdf.body_text("Duration: 36 months (3 years)")
    pdf.body_text(
        "Phase 1 (Months 1–12): Sensor design, prototyping, and lab validation.\n"
        "Phase 2 (Months 13–24): Field deployment in two partner mines.\n"
        "Phase 3 (Months 25–36): Data analysis, system refinement, and reporting."
    )

    pdf.section_title("6. References")
    pdf.body_text(
        "1. Bureau of Indian Standards, IS 6069:2016 – Safety in Mines.\n"
        "2. DGMS Technical Circular No. 5, 2023 – Methane Management.\n"
        "3. IEA Clean Coal Centre, Report CCC/298, 2022."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(output_path: str = "sample_proposal.pdf") -> str:
    """Build the sample PDF and return its absolute path."""
    pdf = ProposalPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    _page_cover(pdf)
    _page_summary(pdf)
    _page_budget(pdf)

    pdf.output(output_path)
    abs_path = os.path.abspath(output_path)
    return abs_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sample_proposal.pdf"
    path = generate(out)
    print(f"[OK] Sample proposal PDF created -> {path}")
    print(f"     Pages : 3")
    print(f"     Size  : {os.path.getsize(path) / 1024:.1f} KB")

