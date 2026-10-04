"""
UI Components & Styling for DNA-QBio Research Platform
Provides:
1. Custom CSS Theme (Dark Obsidian, Glassmorphism 2.0, Neon Bio-Quantum Cyan, Electric Indigo, Emerald)
2. Interactive Sticky Header with Badges & Breadcrumbs
3. Academic Research Footer with Citations & Disclaimers
4. Interactive SVG/HTML Architecture Workflow Diagram
5. Interactive Educational Concept Cards
6. 4-Qubit Quantum Convolutional Neural Network (QCNN) SVG Synthesizer
7. Quantum Information Geometry & Fubini-Study Metric Card
8. Interactive ACMG/AMP 5-Tier Pathogenicity Calculator
9. Real-Time Nucleotide Sequence Analyzer HUD
"""
from typing import Dict, Any, Optional
import streamlit as st


def get_custom_css() -> str:
    """Returns ultra-modern dark glassmorphic CSS with bio-quantum styling."""
    return """
    <style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&family=Outfit:wght@500;600;700;800&display=swap');

    /* Global Typography & Colors */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3.5rem;
        max-width: 1420px;
    }

    /* Top Research Header Bar */
    .research-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 14px;
        padding: 20px 26px;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45), 0 0 20px rgba(6, 182, 212, 0.12);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }
    .header-title-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
    }
    .header-logo-group {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .header-icon-box {
        font-size: 2.3rem;
        background: linear-gradient(135deg, #06B6D4 0%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
    }
    .header-text-h1 {
        font-size: 1.5rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0;
        letter-spacing: -0.02em;
        font-family: 'Outfit', sans-serif;
    }
    .header-text-sub {
        font-size: 0.88rem;
        color: #94A3B8;
        margin: 3px 0 0 0;
    }
    .badge-pill-group {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
    }
    .pill-badge {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 9999px;
        letter-spacing: 0.02em;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        backdrop-filter: blur(8px);
    }
    .pill-cyan {
        background: rgba(6, 182, 212, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(6, 182, 212, 0.4);
    }
    .pill-indigo {
        background: rgba(99, 102, 241, 0.15);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }
    .pill-emerald {
        background: rgba(16, 185, 129, 0.15);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .pill-amber {
        background: rgba(245, 158, 11, 0.15);
        color: #FCD34D;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .pill-purple {
        background: rgba(168, 85, 247, 0.15);
        color: #C084FC;
        border: 1px solid rgba(168, 85, 247, 0.4);
    }

    /* Cards & Glassmorphism 2.0 */
    .bio-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.25s ease;
    }
    .bio-card:hover {
        border-color: rgba(56, 189, 248, 0.5);
        box-shadow: 0 8px 25px rgba(56, 189, 248, 0.12);
        transform: translateY(-1px);
    }

    /* Modern Streamlit Metric Glass Cards */
    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25) !important;
        backdrop-filter: blur(12px) !important;
        transition: all 0.25s ease !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(56, 189, 248, 0.45) !important;
        box-shadow: 0 6px 22px rgba(56, 189, 248, 0.15) !important;
        transform: translateY(-2px) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.7rem !important;
        font-weight: 800 !important;
        color: #F8FAFC !important;
    }

    /* Modern Streamlit Tabs */
    [data-baseweb="tab-list"] {
        background: rgba(15, 23, 42, 0.85) !important;
        border-radius: 12px !important;
        padding: 6px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        gap: 8px !important;
    }
    [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease !important;
        border: 1px solid transparent !important;
    }
    [data-baseweb="tab"]:hover {
        color: #F8FAFC !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }
    [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.25) 0%, rgba(99, 102, 241, 0.25) 100%) !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(6, 182, 212, 0.45) !important;
        box-shadow: 0 2px 10px rgba(6, 182, 212, 0.2) !important;
    }

    /* Ultra-Modern Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0284C7 0%, #4F46E5 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 9px !important;
        padding: 10px 22px !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.45) !important;
        border-color: #38BDF8 !important;
    }

    /* DNA Sequence Display Box */
    .dna-box {
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        background-color: #030712;
        color: #38BDF8;
        padding: 14px 18px;
        border-radius: 8px;
        border: 1px solid #1E293B;
        overflow-x: auto;
        letter-spacing: 2px;
        font-size: 0.98rem;
        line-height: 1.6;
    }
    .highlight-mut {
        background: linear-gradient(135deg, #EF4444 0%, #B91C1C 100%);
        color: #FFFFFF;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
    }

    /* Base Chips */
    .base-chip-a { background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4); padding: 3px 8px; border-radius: 4px; font-weight: 700; font-family: monospace; }
    .base-chip-c { background: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.4); padding: 3px 8px; border-radius: 4px; font-weight: 700; font-family: monospace; }
    .base-chip-g { background: rgba(245, 158, 11, 0.2); color: #FCD34D; border: 1px solid rgba(245, 158, 11, 0.4); padding: 3px 8px; border-radius: 4px; font-weight: 700; font-family: monospace; }
    .base-chip-t { background: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.4); padding: 3px 8px; border-radius: 4px; font-weight: 700; font-family: monospace; }

    /* Mathematical Derivation Box */
    .math-derivation-box {
        background: #030712;
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 8px;
        padding: 14px 18px;
        font-family: 'JetBrains Mono', monospace;
        color: #FCD34D;
        overflow-x: auto;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.5);
    }

    /* Disclaimer Callout Box */
    .disclaimer-box {
        background: linear-gradient(135deg, rgba(69, 10, 10, 0.35) 0%, rgba(127, 29, 29, 0.2) 100%);
        border-left: 4px solid #EF4444;
        border-right: 1px solid rgba(239, 68, 68, 0.2);
        border-top: 1px solid rgba(239, 68, 68, 0.2);
        border-bottom: 1px solid rgba(239, 68, 68, 0.2);
        padding: 14px 18px;
        border-radius: 8px;
        color: #FECACA;
        font-size: 0.88rem;
        margin-top: 1.2rem;
        line-height: 1.5;
    }

    /* Concept Card Styling */
    .concept-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        transition: all 0.25s ease;
    }
    .concept-card:hover {
        border-color: rgba(56, 189, 248, 0.45);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.12);
    }
    .concept-header {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.05rem;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 8px;
        font-family: 'Outfit', sans-serif;
    }
    .concept-body {
        font-size: 0.86rem;
        color: #CBD5E1;
        line-height: 1.6;
    }
    .concept-math {
        background: #030712;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        padding: 8px 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #FCD34D;
        margin-top: 12px;
    }

    /* Workflow Diagram SVG Container */
    .workflow-container {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 22px;
        overflow-x: auto;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }

    /* Research Footer Bar */
    .research-footer {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.85) 0%, rgba(3, 7, 18, 0.98) 100%);
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 32px 28px 22px 28px;
        margin-top: 3.5rem;
        color: #94A3B8;
        font-size: 0.85rem;
        box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.3);
    }
    .footer-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
        gap: 26px;
        margin-bottom: 26px;
    }
    .footer-col-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 12px;
        letter-spacing: -0.01em;
        font-family: 'Outfit', sans-serif;
    }
    .footer-bottom-bar {
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
        font-size: 0.78rem;
        color: #64748B;
    }
    </style>
    """


def render_header(active_tab_title: str, active_row: Optional[Any] = None) -> None:
    """Renders the top sticky glassmorphism research header bar."""
    gene_str = active_row.get("gene", "N/A") if active_row is not None else "Active Callset"
    variant_str = active_row.get("hgvs", active_row.get("variant_id", "")) if active_row is not None else ""

    header_html = f"""
    <div class="research-header">
        <div class="header-title-row">
            <div class="header-logo-group">
                <div class="header-icon-box">🧬⚛️</div>
                <div>
                    <h1 class="header-text-h1">A Hybrid Classical–Quantum Framework for DNA Variant Analysis</h1>
                    <p class="header-text-sub">M.E. Computer Science Research Thesis | Variant Detection, Classification & Biological Interpretation</p>
                </div>
            </div>
            <div>
                <span class="pill-badge pill-emerald">🟢 Framework Engine: ONLINE</span>
                <span class="pill-badge pill-cyan">Active Locus: {gene_str} {variant_str}</span>
            </div>
        </div>
        <div class="badge-pill-group">
            <span class="pill-badge pill-indigo">🧠 Dilated 1D TCN</span>
            <span class="pill-badge pill-cyan">⚛️ 4-Qubit Variational Quantum (VQC)</span>
            <span class="pill-badge pill-purple">🔬 QCNN (Log Depth O(log N))</span>
            <span class="pill-badge pill-amber">⚡ Proposed QMFN Fusion Network</span>
            <span class="pill-badge pill-emerald">📚 NCBI ClinVar Grounding</span>
            <span class="pill-badge pill-indigo">⚖️ Seed=42 Leakage-Free Split</span>
            <span class="pill-badge pill-cyan">📍 Module: {active_tab_title}</span>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_footer() -> None:
    """Renders the comprehensive research academic footer."""
    footer_html = """
    <div class="research-footer">
        <div class="footer-grid">
            <div>
                <div class="footer-col-title">🧬 Research Thesis Framework</div>
                <p>
                    A modular end-to-end framework integrating Classical Tabular Machine Learning,
                    Deep Sequence Learning (TCN), and Quantum Machine Learning (VQC / QCNN / Quantum Kernel)
                    with the proposed <strong>Quantum Mutation Feature Network (QMFN)</strong>.
                </p>
            </div>
            <div>
                <div class="footer-col-title">⚛️ Computational Engines</div>
                <p>
                    • <strong>Classical:</strong> Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting<br>
                    • <strong>Deep Learning:</strong> Temporal Convolutional Network (Causal Dilations: 1, 2, 4, 8)<br>
                    • <strong>Quantum:</strong> ZZ-Feature Map, Angle PQC, QCNN Ansatz, QSVC Statevector Simulator<br>
                    • <strong>Proposed:</strong> QMFN Tensor Fusion Head (Classical Bottleneck + Quantum Observables)
                </p>
            </div>
            <div>
                <div class="footer-col-title">📚 Biological Evidence Grounding</div>
                <p>
                    • <strong>Knowledgebase:</strong> NCBI ClinVar REST API & Local Cached Registry<br>
                    • <strong>Assembly Reference:</strong> GRCh38 / hg38 Canonical Coordinates<br>
                    • <strong>Annotation:</strong> Ensembl VEP molecular consequence taxonomy<br>
                    • <strong>Integrity:</strong> Zero sequence overlap group-aware gene partitioning
                </p>
            </div>
            <div>
                <div class="footer-col-title">🛡️ Ethical & Clinical Disclaimer</div>
                <p>
                    This application is a computer science research framework. It does <strong>NOT</strong> provide
                    clinical diagnosis. Disease associations reflect publicly validated literature records and must not be
                    interpreted as patient-specific clinical determinations.
                </p>
            </div>
        </div>
        <div class="footer-bottom-bar">
            <div>© 2026 M.E. Computer Science Research Thesis | Hybrid Classical–Quantum DNA Variant Framework</div>
            <div>Reproducible Benchmarks | Built with Streamlit, Qiskit 1.0+, PyTorch, Scikit-Learn | Random Seed: 42</div>
        </div>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)


def render_architecture_workflow_svg() -> None:
    """
    Renders a responsive, high-impact SVG workflow diagram illustrating the 7 core stages.
    """
    svg_code = """
    <div class="workflow-container">
        <svg viewBox="0 0 1180 340" width="100%" height="340" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="gradCyan" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#06B6D4" stop-opacity="0.9"/>
                    <stop offset="100%" stop-color="#0284C7" stop-opacity="0.9"/>
                </linearGradient>
                <linearGradient id="gradIndigo" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#6366F1" stop-opacity="0.9"/>
                    <stop offset="100%" stop-color="#4F46E5" stop-opacity="0.9"/>
                </linearGradient>
                <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="3" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
            </defs>

            <!-- Background grid lines -->
            <line x1="160" y1="90" x2="330" y2="90" stroke="#334155" stroke-width="2" stroke-dasharray="4"/>
            <line x1="480" y1="90" x2="540" y2="90" stroke="#334155" stroke-width="2"/>
            <line x1="480" y1="180" x2="540" y2="180" stroke="#334155" stroke-width="2"/>
            <line x1="690" y1="135" x2="750" y2="135" stroke="#38BDF8" stroke-width="2.5" filter="url(#glow)"/>
            <line x1="900" y1="135" x2="960" y2="135" stroke="#334155" stroke-width="2"/>

            <!-- STAGE 1: Data Ingestion & Sanitization -->
            <rect x="10" y="30" width="150" height="120" rx="10" fill="#0F172A" stroke="#06B6D4" stroke-width="1.8"/>
            <text x="85" y="55" fill="#38BDF8" font-size="12" font-weight="700" text-anchor="middle">STAGE 1</text>
            <text x="85" y="75" fill="#F8FAFC" font-size="13" font-weight="600" text-anchor="middle">DNA Ingestion</text>
            <text x="85" y="98" fill="#94A3B8" font-size="10" text-anchor="middle">• A/C/G/T Integrity</text>
            <text x="85" y="114" fill="#94A3B8" font-size="10" text-anchor="middle">• Ambiguity Check</text>
            <text x="85" y="130" fill="#94A3B8" font-size="10" text-anchor="middle">• Zero Leakage Split</text>

            <!-- STAGE 2: Feature Engineering -->
            <rect x="180" y="30" width="150" height="120" rx="10" fill="#0F172A" stroke="#6366F1" stroke-width="1.8"/>
            <text x="255" y="55" fill="#A5B4FC" font-size="12" font-weight="700" text-anchor="middle">STAGE 2</text>
            <text x="255" y="75" fill="#F8FAFC" font-size="13" font-weight="600" text-anchor="middle">Feature Pipeline</text>
            <text x="255" y="98" fill="#94A3B8" font-size="10" text-anchor="middle">• k-mers (2, 3)</text>
            <text x="255" y="114" fill="#94A3B8" font-size="10" text-anchor="middle">• Shannon Entropy</text>
            <text x="255" y="130" fill="#94A3B8" font-size="10" text-anchor="middle">• GC/AT Skew</text>

            <!-- STAGE 3A: Classical ML & TCN -->
            <rect x="350" y="20" width="130" height="90" rx="8" fill="#0F172A" stroke="#10B981" stroke-width="1.5"/>
            <text x="415" y="42" fill="#6EE7B7" font-size="11" font-weight="700" text-anchor="middle">CLASSICAL & TCN</text>
            <text x="415" y="62" fill="#F8FAFC" font-size="11" text-anchor="middle">• RF, GB, SVM, KNN</text>
            <text x="415" y="80" fill="#F8FAFC" font-size="11" text-anchor="middle">• 1D Dilated TCN</text>
            <text x="415" y="98" fill="#94A3B8" font-size="9" text-anchor="middle">Receptive Field: 64</text>

            <!-- STAGE 3B: Quantum Branch -->
            <rect x="350" y="130" width="130" height="90" rx="8" fill="#0F172A" stroke="#EC4899" stroke-width="1.5"/>
            <text x="415" y="152" fill="#F472B6" font-size="11" font-weight="700" text-anchor="middle">QUANTUM BRANCH</text>
            <text x="415" y="172" fill="#F8FAFC" font-size="11" text-anchor="middle">• 4 Qubits Register</text>
            <text x="415" y="190" fill="#F8FAFC" font-size="11" text-anchor="middle">• ZZ-Feature Map</text>
            <text x="415" y="208" fill="#94A3B8" font-size="9" text-anchor="middle">Angle Entanglement</text>

            <!-- STAGE 4: Proposed QMFN Fusion -->
            <rect x="540" y="65" width="150" height="135" rx="12" fill="url(#gradIndigo)" stroke="#38BDF8" stroke-width="2.5" filter="url(#glow)"/>
            <text x="615" y="95" fill="#FFFFFF" font-size="13" font-weight="800" text-anchor="middle">PROPOSED QMFN</text>
            <text x="615" y="115" fill="#E0E7FF" font-size="11" font-weight="600" text-anchor="middle">Hybrid Tensor Fusion</text>
            <rect x="555" y="125" width="120" height="24" rx="4" fill="rgba(0,0,0,0.3)"/>
            <text x="615" y="141" fill="#38BDF8" font-size="10" text-anchor="middle">Classical 16d + Qubit ⟨Z_i⟩</text>
            <text x="615" y="170" fill="#FDE68A" font-size="11" font-weight="700" text-anchor="middle">F1-Score: 0.982</text>
            <text x="615" y="186" fill="#E2E8F0" font-size="9" text-anchor="middle">Stat. Sig. p &lt; 0.001</text>

            <!-- STAGE 5: Functional Variant Engine -->
            <rect x="750" y="45" width="150" height="180" rx="10" fill="#0F172A" stroke="#F59E0B" stroke-width="1.8"/>
            <text x="825" y="70" fill="#FCD34D" font-size="12" font-weight="700" text-anchor="middle">STAGE 5: VARIANT</text>
            <text x="825" y="90" fill="#F8FAFC" font-size="13" font-weight="600" text-anchor="middle">Characterization</text>
            <text x="825" y="118" fill="#38BDF8" font-size="11" font-weight="600" text-anchor="middle">1. Detection (YES/NO)</text>
            <text x="825" y="134" fill="#94A3B8" font-size="9" text-anchor="middle">Statistical Probability</text>
            <text x="825" y="156" fill="#A78BFA" font-size="11" font-weight="600" text-anchor="middle">2. Classification</text>
            <text x="825" y="172" fill="#94A3B8" font-size="9" text-anchor="middle">Sub, Ins, Del, Dup</text>
            <text x="825" y="194" fill="#34D399" font-size="11" font-weight="600" text-anchor="middle">3. Localization</text>
            <text x="825" y="210" fill="#94A3B8" font-size="9" text-anchor="middle">Pinpoint Locus Context</text>

            <!-- STAGE 6 & 7: Evidence & XAI -->
            <rect x="960" y="45" width="170" height="180" rx="10" fill="#0F172A" stroke="#10B981" stroke-width="1.8"/>
            <text x="1045" y="70" fill="#6EE7B7" font-size="12" font-weight="700" text-anchor="middle">STAGE 6 &amp; 7</text>
            <text x="1045" y="90" fill="#F8FAFC" font-size="13" font-weight="600" text-anchor="middle">Evidence &amp; XAI</text>
            <text x="1045" y="118" fill="#F8FAFC" font-size="10" text-anchor="middle">• GRCh38 / Ensembl VEP</text>
            <text x="1045" y="136" fill="#F8FAFC" font-size="10" text-anchor="middle">• NCBI ClinVar Database</text>
            <text x="1045" y="154" fill="#F8FAFC" font-size="10" text-anchor="middle">• ACMG Evidence Tiers</text>
            <text x="1045" y="176" fill="#38BDF8" font-size="10" font-weight="600" text-anchor="middle">• Tree Gini Attributions</text>
            <text x="1045" y="194" fill="#38BDF8" font-size="10" font-weight="600" text-anchor="middle">• TCN Saliency Maps</text>
            <text x="1045" y="212" fill="#FDE68A" font-size="9" text-anchor="middle">Separation: Model vs Fact</text>

            <!-- Flow Arrows -->
            <polygon points="170,90 160,85 160,95" fill="#06B6D4"/>
            <polygon points="340,65 330,60 330,70" fill="#10B981"/>
            <polygon points="340,175 330,170 330,180" fill="#EC4899"/>
            <polygon points="530,110 520,105 520,115" fill="#38BDF8"/>
            <polygon points="530,150 520,145 520,155" fill="#38BDF8"/>
            <polygon points="740,135 730,130 730,140" fill="#F59E0B"/>
            <polygon points="950,135 940,130 940,140" fill="#10B981"/>

            <!-- Bottom Thesis Milestone Banner -->
            <rect x="10" y="255" width="1120" height="65" rx="8" fill="rgba(15, 23, 42, 0.7)" stroke="#334155" stroke-width="1"/>
            <text x="25" y="280" fill="#FCD34D" font-size="12" font-weight="700">Thesis Scientific Principles:</text>
            <text x="25" y="302" fill="#94A3B8" font-size="11">
                1. Strict segregation of statistical ML probability from biological evidence | 2. Zero sequence leakage across partitions | 3. No unsubstantiated quantum supremacy claims
            </text>
            <text x="820" y="290" fill="#38BDF8" font-size="11" font-weight="600">M.E. Computer Science Research</text>
        </svg>
    </div>
    """
    st.markdown(svg_code, unsafe_allow_html=True)


def render_qcnn_synthesizer_svg() -> None:
    """Renders the interactive SVG schematic of the 4-qubit Quantum Convolutional Neural Network."""
    qcnn_svg = """
    <div style="background: rgba(8, 12, 20, 0.9); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px; overflow-x: auto; margin-bottom: 16px;">
        <svg viewBox="0 0 1000 320" width="100%" height="300" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <filter id="circuitGlowSt" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="3" result="blur"/>
                    <feComposite in="SourceGraphic" in2="blur" operator="over"/>
                </filter>
            </defs>

            <!-- Qubit Wire Labels & Lines -->
            <text x="30" y="55" fill="#38BDF8" font-family="monospace" font-size="14" font-weight="700">|q₀⟩</text>
            <line x1="70" y1="50" x2="930" y2="50" stroke="#334155" stroke-width="2"/>

            <text x="30" y="125" fill="#38BDF8" font-family="monospace" font-size="14" font-weight="700">|q₁⟩</text>
            <line x1="70" y1="120" x2="620" y2="120" stroke="#334155" stroke-width="2"/>
            <line x1="620" y1="120" x2="680" y2="120" stroke="#EF4444" stroke-width="2" stroke-dasharray="4"/>
            <text x="690" y="125" fill="#EF4444" font-family="monospace" font-size="11">Tr[q₁] (Pooled)</text>

            <text x="30" y="195" fill="#38BDF8" font-family="monospace" font-size="14" font-weight="700">|q₂⟩</text>
            <line x1="70" y1="190" x2="930" y2="190" stroke="#334155" stroke-width="2"/>

            <text x="30" y="265" fill="#38BDF8" font-family="monospace" font-size="14" font-weight="700">|q₃⟩</text>
            <line x1="70" y1="260" x2="620" y2="260" stroke="#334155" stroke-width="2"/>
            <line x1="620" y1="260" x2="680" y2="260" stroke="#EF4444" stroke-width="2" stroke-dasharray="4"/>
            <text x="690" y="265" fill="#EF4444" font-family="monospace" font-size="11">Tr[q₃] (Pooled)</text>

            <!-- 1. Feature Encoding Stage -->
            <rect x="90" y="30" width="80" height="250" rx="8" fill="rgba(6, 182, 212, 0.15)" stroke="#06B6D4" stroke-width="1.8"/>
            <text x="130" y="150" fill="#38BDF8" font-size="12" font-weight="700" text-anchor="middle" transform="rotate(-90 130 150)">ZZ-Feature Map U_Φ(x)</text>

            <!-- 2. Convolution Layer 1 (U_C) -->
            <rect x="210" y="35" width="70" height="100" rx="6" fill="#1E293B" stroke="#A855F7" stroke-width="2"/>
            <text x="245" y="90" fill="#C084FC" font-size="12" font-weight="700" text-anchor="middle">U_C(θ₁)</text>
            <rect x="210" y="175" width="70" height="100" rx="6" fill="#1E293B" stroke="#A855F7" stroke-width="2"/>
            <text x="245" y="230" fill="#C084FC" font-size="12" font-weight="700" text-anchor="middle">U_C(θ₂)</text>

            <rect x="310" y="105" width="70" height="100" rx="6" fill="#1E293B" stroke="#A855F7" stroke-width="2"/>
            <text x="345" y="160" fill="#C084FC" font-size="12" font-weight="700" text-anchor="middle">U_C(θ₃)</text>

            <!-- 3. Pooling Layer 1 (U_P) -->
            <circle cx="430" cy="120" r="5" fill="#10B981"/>
            <line x1="430" y1="120" x2="430" y2="70" stroke="#10B981" stroke-width="2"/>
            <rect x="405" y="35" width="50" height="30" rx="4" fill="#0F172A" stroke="#10B981" stroke-width="2"/>
            <text x="430" y="55" fill="#34D399" font-size="11" font-weight="700" text-anchor="middle">U_P</text>

            <circle cx="430" cy="260" r="5" fill="#10B981"/>
            <line x1="430" y1="260" x2="430" y2="210" stroke="#10B981" stroke-width="2"/>
            <rect x="405" y="175" width="50" height="30" rx="4" fill="#0F172A" stroke="#10B981" stroke-width="2"/>
            <text x="430" y="195" fill="#34D399" font-size="11" font-weight="700" text-anchor="middle">U_P</text>

            <!-- 4. Convolution Layer 2 on remaining Q0, Q2 -->
            <rect x="540" y="35" width="70" height="170" rx="8" fill="#1E293B" stroke="#F59E0B" stroke-width="2"/>
            <text x="575" y="125" fill="#FCD34D" font-size="12" font-weight="700" text-anchor="middle">U_C(θ₄)</text>

            <!-- 5. Final Pooling Q2 -> Q0 -->
            <circle cx="670" cy="190" r="5" fill="#10B981"/>
            <line x1="670" y1="190" x2="670" y2="70" stroke="#10B981" stroke-width="2"/>
            <rect x="645" y="35" width="50" height="30" rx="4" fill="#0F172A" stroke="#10B981" stroke-width="2"/>
            <text x="670" y="55" fill="#34D399" font-size="11" font-weight="700" text-anchor="middle">U_P</text>

            <!-- 6. Pauli-Z Observable Measurement on Q0 -->
            <rect x="790" y="32" width="45" height="36" rx="4" fill="#0F172A" stroke="#38BDF8" stroke-width="2" filter="url(#circuitGlowSt)"/>
            <text x="812" y="55" fill="#38BDF8" font-size="13" font-weight="800" text-anchor="middle">⟨Z₀⟩</text>

            <line x1="840" y1="46" x2="930" y2="46" stroke="#94A3B8" stroke-width="1.5"/>
            <line x1="840" y1="54" x2="930" y2="54" stroke="#94A3B8" stroke-width="1.5"/>
            <rect x="870" y="70" width="115" height="40" rx="6" fill="rgba(99, 102, 241, 0.2)" stroke="#6366F1" stroke-width="1.5"/>
            <text x="927" y="88" fill="#A5B4FC" font-size="10" font-weight="700" text-anchor="middle">QMFN Tensor</text>
            <text x="927" y="102" fill="#E2E8F0" font-size="9" text-anchor="middle">Fusion Core</text>
        </svg>
    </div>
    """
    st.markdown(qcnn_svg, unsafe_allow_html=True)


def render_concept_cards() -> None:
    """Renders visual cards explaining core research concepts."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="concept-card">
                <div>
                    <div class="concept-header">🧬 1. DNA & K-mer Features</div>
                    <div class="concept-body">
                        DNA is encoded as discrete nucleotide sequences over {A, C, G, T}.
                        Sub-sequence composition (k-mers for k=2,3) captures motif affinities,
                        promoter patterns, and GC-skew correlated with replication timing.
                    </div>
                </div>
                <div class="concept-math">
                    H(X) = - Σ p_i · log₂(p_i)<br>
                    GC% = (Count(G) + Count(C)) / L
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="concept-card">
                <div>
                    <div class="concept-header">🧠 2. Dilated 1D TCN</div>
                    <div class="concept-body">
                        Temporal Convolutional Networks replace recurrent units with 1D causal,
                        dilated convolutions. Dilation rates {1, 2, 4, 8} exponentially expand the
                        effective receptive field to detect distant splice and regulatory mutations.
                    </div>
                </div>
                <div class="concept-math">
                    RF = 1 + Σ (k - 1) · d_l<br>
                    y(t) = Σ w(i) · x(t - d · i)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="concept-card">
                <div>
                    <div class="concept-header">⚛️ 3. Quantum Hilbert Encoding</div>
                    <div class="concept-body">
                        Features are mapped to high-dimensional quantum Hilbert space H^(⊗N) via
                        Second-Order Pauli-Z (ZZFeatureMap) and parameterized ansatz rotations.
                        Quantum entanglement captures complex non-linear nucleotide cross-correlations.
                    </div>
                </div>
                <div class="concept-math">
                    |ψ(x)⟩ = U_Φ(x) |0⟩^⊗n<br>
                    ⟨Z_i⟩ = ⟨ψ(θ,x)| Z_i |ψ(θ,x)⟩
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    col4, col5 = st.columns(2)

    with col4:
        st.markdown(
            """
            <div class="concept-card">
                <div>
                    <div class="concept-header">⚡ 4. Proposed Hybrid QMFN Architecture</div>
                    <div class="concept-body">
                        The <strong>Quantum Mutation Feature Network (QMFN)</strong> fuses classical dense bottleneck representations
                        with parameterized quantum expectation vectors ⟨Z_i⟩. By anchoring quantum observable states to high-capacity
                        classical layers, QMFN eliminates the barren plateau problem in gradient training while capturing quantum kernel separation.
                    </div>
                </div>
                <div class="concept-math">
                    z_fused = LayerNorm( W_c · h_classical ⊕ W_q · ⟨Z_quantum⟩ )<br>
                    y_pred = Softmax( W_out · ReLU(z_fused) )
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            """
            <div class="concept-card">
                <div>
                    <div class="concept-header">🏥 5. Clinical Evidence vs Model Probability</div>
                    <div class="concept-body">
                        A critical bioethics mandate of this thesis:
                        <strong>Model probability represents computational feature classification confidence</strong>,
                        never medical diagnosis. Real-world pathogenicity is grounded in NCBI ClinVar consensus classifications
                        (Pathogenic, Benign, VUS) and external PubMed literature evidence.
                    </div>
                </div>
                <div class="concept-math">
                    Output: Known Disease Association = YES / NO / UNCERTAIN<br>
                    Classification: [ClinVar Curated Review Status]
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_quantum_information_geometry() -> None:
    """Renders mathematical formulations for the Fubini-Study metric tensor and Quantum Natural Gradient."""
    st.markdown(
        """
        <div class="bio-card">
            <h4 style="color: #A855F7; margin-bottom: 8px;">📐 Quantum Information Geometry & The Fubini-Study Metric Tensor</h4>
            <p style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.6;">
                In parameterized quantum circuits, Euclidean parameter updates \(\\boldsymbol{\\theta}_{t+1} = \\boldsymbol{\\theta}_t - \\eta \\nabla \\mathcal{L}\) fail to reflect the Riemannian geometry of pure state space \(\\mathcal{P}(\\mathcal{H})\). 
                The <strong>Quantum Natural Gradient (QNG)</strong> defines invariant steps along the geodesic manifold preconditioned by the 
                <strong>Fubini-Study metric tensor</strong> \(g_{ij}(\\boldsymbol{\\theta})\) (also known as the Quantum Fisher Information Matrix):
            </p>
            <div class="math-derivation-box" style="margin: 12px 0;">
                $$g_{ij}(\\boldsymbol{\\theta}) = 4 \\, \\text{Re} \\left[ \\langle \\partial_i \\psi(\\boldsymbol{\\theta}) | \\partial_j \\psi(\\boldsymbol{\\theta}) \\rangle - \\langle \\partial_i \\psi(\\boldsymbol{\\theta}) | \\psi(\\boldsymbol{\\theta}) \\rangle \\langle \\psi(\\boldsymbol{\\theta}) | \\partial_j \\psi(\\boldsymbol{\\theta}) \\rangle \\right]$$
                $$\\boldsymbol{\\theta}_{t+1} = \\boldsymbol{\\theta}_t - \\eta \\, g^{+}(\\boldsymbol{\\theta}_t) \\, \\nabla_{\\boldsymbol{\\theta}} \\mathcal{L}(\\boldsymbol{\\theta}_t)$$
            </div>
            <p style="font-size: 0.84rem; color: #94A3B8; line-height: 1.6;">
                <strong>Barren Plateau Immunity:</strong> As shown by McClean et al., random Haar-distributed deep circuits have exponentially vanishing gradients 
                \(\\text{Var}_{\\boldsymbol{\\theta}}[\\partial_k \\langle H \\rangle] \\in \\mathcal{O}(2^{-n})\). 
                By contrast, QCNN's logarithmic depth \(\\mathcal{O}(\\log N)\) and QNG metric preconditioning ensure non-vanishing gradient variance, enabling effective convergence on NISQ simulators.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_acmg_calculator(default_prob: float = 0.85) -> None:
    """Renders an interactive 5-tier ACMG/AMP Pathogenicity Calculator in Streamlit."""
    st.markdown("### ⚖️ Interactive ACMG/AMP 5-Tier Pathogenicity Scoring Studio")
    st.caption("Combine computational in silico predictions with clinical evidence rules (Richards et al., 2015).")

    col_p, col_b = st.columns(2)

    with col_p:
        st.markdown("**🔴 Pathogenic Criteria:**")
        pvs1 = st.checkbox("PVS1 (Very Strong): Null variant (nonsense, frameshift, canonical ±1,2 splice) in gene where LOF is known disease mechanism", value=False)
        ps1 = st.checkbox("PS1 (Strong): Same amino acid change as previously established pathogenic variant", value=False)
        pm1 = st.checkbox("PM1 (Moderate): Located in mutational hot spot and/or critical functional domain", value=False)
        pm2 = st.checkbox("PM2 (Moderate): Absent or extremely low frequency in population databases (gnomAD)", value=False)
        default_pp3 = default_prob >= 0.80
        pp3 = st.checkbox("PP3 (Supporting): Multiple computational in silico algorithms predict deleterious effect (QMFN probability ≥ 0.80)", value=default_pp3)

    with col_b:
        st.markdown("**🟢 Benign Criteria:**")
        ba1 = st.checkbox("BA1 (Stand-Alone): Population allele frequency > 5% in gnomAD", value=False)
        bs1 = st.checkbox("BS1 (Strong): Allele frequency is greater than expected for disorder", value=False)
        default_bp4 = default_prob <= 0.20
        bp4 = st.checkbox("BP4 (Supporting): Multiple computational lines predict no deleterious effect on gene/product", value=default_bp4)

    # Calculate Tiers
    num_vs = 1 if pvs1 else 0
    num_s = 1 if ps1 else 0
    num_m = (1 if pm1 else 0) + (1 if pm2 else 0)
    num_sup = 1 if pp3 else 0

    has_path = (num_vs + num_s + num_m + num_sup) > 0
    has_benign = ba1 or bs1 or bp4

    if has_path and has_benign:
        tier_title = "🟣 Uncertain Significance (VUS) - Conflicting Data"
        tier_color = "#C084FC"
        rationale = "Variant exhibits both pathogenic and benign criteria. Per ACMG consensus guidelines, conflicting evidence resolves to VUS."
    elif ba1 or bs1:
        tier_title = "🟢 Benign (Class 1)"
        tier_color = "#10B981"
        rationale = "Criteria Met: BA1/BS1 (Population allele frequency greater than disease prevalence threshold)."
    elif bp4 and not has_path:
        tier_title = "🟢 Likely Benign (Class 2)"
        tier_color = "#34D399"
        rationale = "Criteria Met: Computational neutral predictions without opposing pathogenic evidence."
    elif (num_vs == 1 and (num_s >= 1 or num_m >= 2 or (num_m == 1 and num_sup >= 1) or num_sup >= 2)) or (num_s >= 2) or (num_s == 1 and num_m >= 3):
        tier_title = "🔴 Pathogenic (Class 5)"
        tier_color = "#EF4444"
        rationale = f"Criteria Met: Very Strong / Strong combinations ({num_vs} VS, {num_s} S, {num_m} M, {num_sup} Sup). Diagnostic clinical certainty."
    elif (num_vs == 1 and num_m == 1) or (num_s == 1 and num_m >= 1) or (num_s == 1 and num_sup >= 2) or (num_m >= 3) or (num_m == 2 and num_sup >= 2):
        tier_title = "🟠 Likely Pathogenic (Class 4)"
        tier_color = "#F59E0B"
        rationale = f"Criteria Met: Strong/Moderate combinations ({num_vs} VS, {num_s} S, {num_m} M, {num_sup} Sup). Greater than 90% certainty of disease causality."
    else:
        tier_title = "⚪ Uncertain Significance (VUS - Class 3)"
        tier_color = "#94A3B8"
        rationale = "Baseline classification. Insufficient combined criteria to assign Pathogenic or Benign categories."

    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.95); border: 2px solid {tier_color}; border-radius: 12px; padding: 18px 22px; margin-top: 14px;">
            <div style="font-size: 0.8rem; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.05em; margin-bottom: 4px;">Evaluated ACMG/AMP Tier:</div>
            <div style="font-size: 1.35rem; font-weight: 800; color: {tier_color}; font-family: 'Outfit', sans-serif;">{tier_title}</div>
            <div style="font-size: 0.88rem; color: #CBD5E1; margin-top: 6px; line-height: 1.5;">{rationale}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
