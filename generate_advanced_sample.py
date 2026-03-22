from fpdf import FPDF

class ComprehensiveProposal(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100)
        self.cell(0, 10, "OFFICIAL USE ONLY - MINISTRY OF COAL R&D DIVISION", 0, 1, "R")
        self.ln(5)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 51, 102) # Dark Blue
        self.cell(0, 10, title, "B", 1, "L")
        self.ln(5)

    def add_passage(self, text):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(0)
        self.multi_cell(0, 7, text)
        self.ln(3)

# Initialize PDF
pdf = ComprehensiveProposal()
pdf.add_page()

# --- SECTION 1: COVER ---
pdf.set_font("Helvetica", "B", 18)
pdf.multi_cell(0, 15, "PROPOSAL: INTEGRATED CARBON CAPTURE AND MINE SAFETY MONITORING SYSTEMS", 0, "C")
pdf.ln(10)

# --- SECTION 2: EXECUTIVE SUMMARY (Passage 1) ---
pdf.section_title("1. Executive Summary")
pdf.add_passage(
    "This research initiative seeks to revolutionize the current coal mining landscape by integrating "
    "cutting-edge Carbon Capture technologies with traditional excavation methods. The primary goal is "
    "to reduce the carbon footprint of national mining operations while simultaneously increasing output "
    "efficiency through the use of automated data streams."
)

# --- SECTION 3: PRINCIPAL INVESTIGATOR (Passage 2) ---
pdf.section_title("2. Personnel and Leadership")
pdf.add_passage(
    "The project will be spearheaded by the Principal Investigator, Dr. Vikram Seth. Dr. Seth holds a "
    "Doctorate in Geo-Technical Engineering and has spent 20 years studying the tectonic shifts in "
    "underground mines. He is currently a Senior Fellow at the National Institute of Mining Research."
)

# --- SECTION 4: COAL GASIFICATION (Passage 3 - Keyword 1) ---
pdf.section_title("3. Technical Methodology: Gasification")
pdf.add_passage(
    "A core component of this study is Coal Gasification. Unlike traditional combustion, our method "
    "converts coal into syngas, which can then be used for Blue Hydrogen production. This process is "
    "essential for meeting the 2026 sustainability targets set by the Ministry and ensures that "
    "deep-seated coal reserves are not wasted."
)

# --- SECTION 5: SAFETY PROTOCOLS (Passage 4 - Keyword 2) ---
pdf.section_title("4. Mine Safety Monitoring")
pdf.add_passage(
    "To protect the workforce, we are implementing a proprietary Mine Safety Monitoring framework. "
    "This system utilizes localized sensors to detect seismic vibrations and gas leaks. By focusing "
    "on real-time telemetry, we can provide immediate warnings to personnel in the event of structural "
    "instability or ventilation failure."
)

# --- SECTION 6: FAULT PREDICTION (Passage 5 - Keyword 3) ---
pdf.section_title("5. Geological Analysis")
pdf.add_passage(
    "The research team has developed a new model for Fault Prediction. In deep-shaft mining, "
    "identifying geological faults 2m to 5m ahead of the face is critical. Our algorithm analyzes "
    "historical data to predict where faults will occur, significantly reducing the risk of "
    "unplanned roof collapses."
)

# --- SECTION 7: WASTE TO WEALTH (Passage 6 - Keyword 4) ---
pdf.section_title("6. Environmental Impact")
pdf.add_passage(
    "In alignment with the Waste to Wealth initiative, the project includes a sub-study on utilizing "
    "fly ash for structural brick manufacturing. Furthermore, the plan addresses Pit Lake Management "
    "by converting abandoned mining pits into sustainable fish hatcheries and community water reservoirs."
)

# --- SECTION 8: BUDGETARY REQUIREMENTS (Passage 7) ---
pdf.section_title("7. Financial Estimates")
pdf.add_passage(
    "The financial framework is designed for maximum efficiency. The Total Budget requested for this "
    "three-phase operation is Rs. 42,00,000. This includes equipment procurement, site labor costs, "
    "and computational overhead for the predictive algorithms."
)

# --- SECTION 9: TIMELINE (Passage 8) ---
pdf.section_title("8. Project Duration")
pdf.add_passage(
    "The anticipated Project Duration is 18 months. The first six months will focus on site "
    "preparation, followed by eight months of active testing, and a final four-month period for "
    "data synthesis and reporting to the Central Mine Planning and Design Institute."
)

# --- SECTION 10: ALTERNATIVE ENERGY (Passage 9 - Keyword 5) ---
pdf.section_title("9. Future Scope: Alternative Energy")
pdf.add_passage(
    "Future iterations of this project will explore the integration of Perovskite Solar cells to "
    "power the underground sensor network. Additionally, we are investigating Blue Hydrogen "
    "applications to provide clean fuel for heavy mining machinery on-site."
)

# --- SECTION 11: CONCLUSION (Passage 10) ---
pdf.section_title("10. Conclusion")
pdf.add_passage(
    "In conclusion, this proposal offers a comprehensive solution to the twin challenges of safety "
    "and sustainability. By combining Coal Gasification with advanced safety tech, we are ensuring "
    "the long-term viability of the Indian coal sector."
)

# Output
pdf.output("advanced_proposal.pdf")
print("Success: 'advanced_proposal.pdf' generated with 10 technical passages.")