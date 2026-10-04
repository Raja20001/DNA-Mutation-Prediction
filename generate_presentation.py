"""
DNA-QBio Platform: High-Impact Anna University M.E. CSE Project Presentation PPT Generator
Generates a 25-slide academic, technically detailed, 16:9 widescreen PowerPoint deck:
  - DNA_QBio_Anna_University_ME_CSE_Presentation.pptx
  - DNA_QBio_Thesis_Presentation.pptx

Strictly adheres to Anna University M.E. Computer Science and Engineering project review guidelines,
presenting the actual implementation, algorithms, 346-D feature pipeline, QMFN architecture,
9-model benchmark results, ClinVar integration, and dual web platform deployments.
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# Bio-Quantum Obsidian Color Palette
BG_COLOR = RGBColor(8, 12, 20)          # #080C14 Dark Obsidian
CARD_BG = RGBColor(15, 23, 42)          # #0F172A Slate Card
CARD_BORDER = RGBColor(51, 65, 85)      # #334155 Border
CYAN_ACCENT = RGBColor(6, 182, 212)     # #06B6D4 Neon Cyan
CYAN_LIGHT = RGBColor(56, 189, 248)     # #38BDF8 Light Cyan
PURPLE_ACCENT = RGBColor(139, 92, 246)  # #8B5CF6 Quantum Violet
GREEN_ACCENT = RGBColor(16, 185, 129)   # #10B981 Emerald
RED_ACCENT = RGBColor(239, 68, 68)      # #EF4444 Crimson
TEXT_WHITE = RGBColor(248, 250, 252)    # #F8FAFC White
TEXT_MUTED = RGBColor(148, 163, 184)    # #94A3B8 Muted Slate
TEXT_LIGHT = RGBColor(203, 213, 225)    # #CBD5E1 Light Slate

IMAGE_HERO = PROJECT_ROOT / "flask_app/static/images/quantum_dna_hero.jpg"
IMAGE_QMFN = PROJECT_ROOT / "flask_app/static/images/qmfn_architecture_3d.jpg"
IMAGE_MUTATION = PROJECT_ROOT / "flask_app/static/images/dna_variant_mutation_3d.jpg"


def set_slide_background(slide):
    """Fill slide with dark obsidian background."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = BG_COLOR


def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER):
    """Add a glassmorphism-styled rectangular card container."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.2)
    else:
        shape.line.fill.background()
    return shape


def add_header(slide, title_text, category_badge="ANNA UNIVERSITY M.E. CSE", subtitle_text=None, slide_num=None):
    """Add standard academic header with category pill badge, footer, and slide number."""
    set_slide_background(slide)

    # Category Pill
    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.35), Inches(3.2), Inches(0.32))
    pill.fill.solid()
    pill.fill.fore_color.rgb = RGBColor(20, 30, 50)
    pill.line.color.rgb = CYAN_ACCENT
    pill.line.width = Pt(1)
    tf_pill = pill.text_frame
    tf_pill.word_wrap = False
    tf_pill.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf_pill.paragraphs[0]
    p.text = category_badge.upper()
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = CYAN_LIGHT
    p.alignment = PP_ALIGN.CENTER

    # Title Box
    tx_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(11.733), Inches(0.65))
    tf = tx_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE

    if subtitle_text:
        p2 = tf.add_paragraph()
        p2.text = subtitle_text
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = TEXT_MUTED

    # Academic Footer
    foot_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.05), Inches(11.733), Inches(0.35))
    tf_foot = foot_box.text_frame
    p_foot = tf_foot.paragraphs[0]
    num_str = f"Slide {slide_num} / 25" if slide_num else ""
    p_foot.text = f"M.E. CSE Project Review 2026–2027 | Rajalakshmi Engineering College (Anna University)     {num_str}"
    p_foot.font.size = Pt(8.5)
    p_foot.font.color.rgb = RGBColor(100, 116, 139)


def build_presentation():
    """Build the complete 25-slide academic presentation deck."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: Title Slide (Academic Credentials + REC & Anna University)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)
    add_card(s1, Inches(0.8), Inches(0.6), Inches(11.733), Inches(6.3), bg_color=CARD_BG, border_color=CYAN_ACCENT)

    if IMAGE_HERO.exists():
        s1.shapes.add_picture(str(IMAGE_HERO), Inches(7.2), Inches(0.9), Inches(5.0), Inches(5.6))

    tx = s1.shapes.add_textbox(Inches(1.2), Inches(0.9), Inches(5.8), Inches(5.6))
    tf = tx.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "ANNA UNIVERSITY :: CHENNAI 600 025"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN_LIGHT

    p = tf.add_paragraph()
    p.text = "A Project Presentation"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = PURPLE_ACCENT
    p.space_after = Pt(10)

    p = tf.add_paragraph()
    p.text = "A Hybrid Classical–Quantum Framework for DNA Variant Detection, Classification, and Evidence-Based Biological Interpretation"
    p.font.size = Pt(21)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE
    p.space_after = Pt(14)

    p = tf.add_paragraph()
    p.text = (
        "• Degree: M.E. Computer Science and Engineering\n"
        "• Department: Department of Computer Science and Engineering\n"
        "• Institution: Rajalakshmi Engineering College (Autonomous)\n"
        "• Affiliation: Anna University, Chennai\n"
        "• Academic Year: 2026–2027\n"
        "• Candidate Name: PG Scholar / Candidate\n"
        "• Register Number: 2116244101001"
    )
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_LIGHT

    s1.notes_slide.notes_text_frame.text = (
        "Respected committee members and examiners, good morning. I am presenting my M.E. Computer Science and Engineering "
        "project entitled 'A Hybrid Classical-Quantum Framework for DNA Variant Detection, Classification, and Evidence-Based "
        "Biological Interpretation'. This project was carried out at the Department of Computer Science and Engineering, "
        "Rajalakshmi Engineering College, affiliated with Anna University."
    )

    # =========================================================================
    # SLIDE 2: Introduction
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Introduction: Computational Genomics & Quantum AI", "Slide 2 :: Introduction", "Domain context, clinical importance, and technological background", 2)

    intro_cards = [
        ("🧬 Project Domain", "Bio-Quantum Informatics & Genomic Machine Learning. Focuses on analyzing high-throughput Next-Generation Sequencing (NGS) DNA data to identify functional genetic mutations.", CYAN_ACCENT),
        ("🌍 Background & Importance", "Human genomes contain ~3.2 billion base pairs. A single nucleotide polymorphism (SNP) or indel in critical oncogenes (TP53, BRCA1, EGFR) can trigger oncogenesis or severe hereditary disorders.", PURPLE_ACCENT),
        ("⚡ Motivation", "Distinguishing true pathogenic mutations from benign polymorphisms remains a bottleneck. Classical algorithms fail to capture non-Euclidean epistatic motif interactions in high-order k-mer spaces.", CYAN_LIGHT),
        ("💡 Proposed Solution", "Quantum Multi-Modal Fusion Network (QMFN) that unifies Dilated Causal Temporal Convolutions (TCN) with 4-qubit Parameterized Quantum Circuits (PQC) and live NCBI ClinVar clinical evidence.", GREEN_ACCENT),
    ]
    for i, (title, body, color) in enumerate(intro_cards):
        col = i % 2
        row = i // 2
        l = Inches(0.8 + col * 5.95)
        t = Inches(1.6 + row * 2.6)
        add_card(s2, l, t, Inches(5.75), Inches(2.4))
        tx = s2.shapes.add_textbox(l + Inches(0.2), t + Inches(0.15), Inches(5.35), Inches(2.1))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = color
        p.space_after = Pt(8)
        p2 = tf.add_paragraph()
        p2.text = body
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_LIGHT

    # =========================================================================
    # SLIDE 3: Problem Statement
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Problem Statement & Research Challenges", "Slide 3 :: Problem Statement", "Specific computational and clinical challenges addressed in this research", 3)

    add_card(s3, Inches(0.8), Inches(1.6), Inches(5.75), Inches(5.2))
    tx1 = s3.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.35), Inches(4.8))
    tf1 = tx1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "⚠️ Real-World Genomic Challenges"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = RED_ACCENT
    p.space_after = Pt(8)
    probs = [
        "Variants of Uncertain Significance (VUS): Over 40% of discovered NGS variants lack decisive clinical classification, delaying critical oncological interventions.",
        "Curse of Dimensionality: Tabular k-mer representations scale exponentially (4^k = 256 for 4-mers), leading to extreme data sparsity in classical Euclidean feature space.",
        "Spatial Grammar Loss: Traditional machine learning ignores 5' to 3' directional regulatory grammar and promoter-enhancer sequence context.",
        "Data Leakage in Evaluation: Standard random cross-validation leaks homologous gene family sequences between training and test sets.",
    ]
    for pt in probs:
        p = tf1.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    add_card(s3, Inches(6.8), Inches(1.6), Inches(5.75), Inches(5.2))
    tx2 = s3.shapes.add_textbox(Inches(7.0), Inches(1.75), Inches(5.35), Inches(4.8))
    tf2 = tx2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "🎯 Specific Problem Addressed"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = CYAN_LIGHT
    p.space_after = Pt(8)
    sol_pts = [
        "Zero-Leakage Chromosome Splitting: Design a deterministic pipeline isolating homologous gene regions across training, validation, and test partitions.",
        "High-Resolution Locus Pinpointing: Execute exact 1-based coordinate localization and transition vs transversion allele classification.",
        "Quantum Hilbert Space Encoding: Project non-linear DNA k-mer interactions into a 2^4 = 16-dimensional quantum state space via ZZ-feature map.",
        "Evidence-Grounded Prediction: Couple statistical model probabilities with live NCBI ClinVar and ACMG/AMP clinical classifications without hallucinatory diagnostic claims.",
    ]
    for pt in sol_pts:
        p = tf2.add_paragraph()
        p.text = "✔ " + pt
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    # =========================================================================
    # SLIDE 4: Existing System
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Existing System: Methods, Technologies & Limitations", "Slide 4 :: Existing System", "Critical appraisal of conventional genomic variant classification tools", 4)

    # Table of Existing Approaches
    table_shape = s4.shapes.add_table(4, 5, Inches(0.8), Inches(1.6), Inches(11.733), Inches(3.2))
    tbl = table_shape.table
    tbl.columns[0].width = Inches(2.2)
    tbl.columns[1].width = Inches(2.4)
    tbl.columns[2].width = Inches(2.3)
    tbl.columns[3].width = Inches(2.4)
    tbl.columns[4].width = Inches(2.433)

    ex_headers = ["Existing Approach", "Methodology / Algorithms", "Technologies Used", "Key Advantages", "Critical Limitations"]
    for j, h in enumerate(ex_headers):
        cell = tbl.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(20, 30, 50)
        p = cell.text_frame.paragraphs[0]
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = CYAN_LIGHT
        p.alignment = PP_ALIGN.CENTER

    ex_rows = [
        ("Alignment Callers (GATK / SAMtools)", "Bayesian Haplotype Assembly, Local de Bruijn graphs", "Java, C++, GATK 4.x, HTSlib", "Standardized clinical reference, reliable on high-coverage WGS", "High false-positive rate on low-complexity repeats; misses regulatory epistasis"),
        ("Classical Tabular ML (Random Forest, SVM)", "Gini Impurity, RBF Kernel on 1-mer/2-mer counts", "Scikit-Learn, Biopython", "Rapid training (< 1s), interpretable feature weights", "Discards spatial 5'->3' grammar; vulnerable to homologous gene leakage"),
        ("Standard 1D CNNs (DeepSEA / DanQ)", "1D Convolutions + Max-Pooling + Bi-LSTM", "TensorFlow, PyTorch, Keras", "Captures local motifs; operates on raw sequence", "Future-token receptive leakage; lacks non-Euclidean quantum feature separation"),
    ]
    for i, row in enumerate(ex_rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i + 1, j)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_BG if i % 2 == 0 else RGBColor(18, 26, 46)
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(9)
            p.font.color.rgb = TEXT_LIGHT

    # Existing System Drawbacks Card
    add_card(s4, Inches(0.8), Inches(5.1), Inches(11.733), Inches(1.7))
    tx_bot = s4.shapes.add_textbox(Inches(1.0), Inches(5.2), Inches(11.333), Inches(1.5))
    tf_bot = tx_bot.text_frame
    tf_bot.word_wrap = True
    p = tf_bot.paragraphs[0]
    p.text = "Key Takeaway: Why Existing Approaches Fall Short"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = RED_ACCENT
    p.space_after = Pt(4)
    p2 = tf_bot.add_paragraph()
    p2.text = (
        "Current systems operate either as rigid heuristic pipelines (GATK) or as flat Euclidean classifiers (SVM, RF). "
        "None possess the capability to map non-linear nucleotide entanglements in high-order Hilbert spaces while maintaining "
        "strict temporal causality (dilated causal convolutions) and anchoring scores to live NCBI ClinVar clinical evidence."
    )
    p2.font.size = Pt(10)
    p2.font.color.rgb = TEXT_LIGHT

    # =========================================================================
    # SLIDE 5: Proposed System
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Proposed System: Quantum Multi-Modal Fusion (QMFN)", "Slide 5 :: Proposed System", "Architectural overview, key contributions, and differentiation from existing tools", 5)

    add_card(s5, Inches(0.8), Inches(1.6), Inches(5.75), Inches(5.2))
    tx1 = s5.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.35), Inches(4.8))
    tf1 = tx1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "🏛️ Proposed Architecture & Concept"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN_LIGHT
    p.space_after = Pt(8)
    prop_pts = [
        "Tri-Engine Synergy: Integrates 5 Classical Ensembles, 1 Deep Dilated Causal TCN, and 2 Quantum Classifiers (VQC, QSVC).",
        "Dilated Causal Temporal Convolutions: 4 residual blocks with dilation factors d in {1, 2, 4, 8} providing an exponential receptive field of 63 bp without future leakage.",
        "4-Qubit Parameterized Quantum Circuit: Maps 346-D biophysical features into 2^4 = 16-dimensional Hilbert space via ZZ-feature map and StronglyEntanglingLayers.",
        "Kronecker Tensor Fusion: Computes outer-product h_TCN (x) <Z>_VQC with learned gating parameter alpha to capture cross-modal feature interactions.",
    ]
    for pt in prop_pts:
        p = tf1.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    add_card(s5, Inches(6.8), Inches(1.6), Inches(5.75), Inches(5.2))
    tx2 = s5.shapes.add_textbox(Inches(7.0), Inches(1.75), Inches(5.35), Inches(4.8))
    tf2 = tx2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "🚀 Key Improvements & Novelty"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT
    p.space_after = Pt(8)
    diff_pts = [
        "Superior Accuracy & AUC: Achieves 96.7% Test Accuracy and 0.990 ROC-AUC, outperforming pure TCN (95.0%, 0.985) and Quantum Kernel (90.0%, 0.945).",
        "Sub-4ms Inference Latency: Delivers 3.80 ms inference speed, making it suitable for high-throughput NGS pipelines.",
        "Probability Calibration: Attains Brier score of 0.082, ensuring model confidence reflects true biological probability.",
        "Dual Web Interface & Dossiers: Fully operational via Flask REST platform (port 5000) and Streamlit research portal (port 8501).",
    ]
    for pt in diff_pts:
        p = tf2.add_paragraph()
        p.text = "✔ " + pt
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    # =========================================================================
    # SLIDE 6: Objectives
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Project Objectives & Scope of Work", "Slide 6 :: Objectives", "Seven technically rigorous objectives executed in this research", 6)

    objectives = [
        ("01", "Problem Analysis & Callset Formulation", "Curate and formalize a balanced benchmark callset of 400 sequences across critical human cancer genes (TP53, BRCA1, EGFR, BRAF, KRAS, CFTR) based on GRCh38.p14.", CYAN_ACCENT),
        ("02", "Zero-Leakage Ingestion & Preprocessing", "Design a deterministic pipeline with strict IUPAC audits, exact duplicate resolution, and gene/chromosome-isolated data partitioning.", PURPLE_ACCENT),
        ("03", "346-D Biophysical Feature Engineering", "Extract 16 dinucleotides, 64 trinucleotides, 256 tetranucleotides, Shannon sequence entropy, Gini-Simpson diversity, and GC-skew.", CYAN_LIGHT),
        ("04", "Tri-Engine Model Suite Implementation", "Develop and evaluate 5 Classical Ensembles, 1 Deep Dilated Causal TCN, and 2 Quantum Classifiers (VQC, Quantum Kernel).", GREEN_ACCENT),
        ("05", "QMFN Hybrid Architecture Formulation", "Design the Quantum Multi-Modal Fusion Network using Kronecker tensor outer-products and learned residual gating.", PURPLE_ACCENT),
        ("06", "Clinical Evidence & XAI Integration", "Integrate live NCBI ClinVar public records, ACMG criteria, Gini feature importance, and nucleotide gradient saliency maps.", CYAN_ACCENT),
        ("07", "Dual-Platform Dashboard Deployment", "Deploy production Flask web application (port 5000) and Streamlit interactive research portal (port 8501) with 3D WebGL visualizations.", GREEN_ACCENT),
    ]

    for i, (num, title, desc, color) in enumerate(objectives):
        col = i % 2 if i < 6 else 0
        row = i // 2 if i < 6 else 3
        w = Inches(5.75) if i < 6 else Inches(11.733)
        l = Inches(0.8 + (col * 5.95 if i < 6 else 0))
        t = Inches(1.6 + row * 1.35)
        h = Inches(1.22)
        add_card(s6, l, t, w, h)
        tx = s6.shapes.add_textbox(l + Inches(0.15), t + Inches(0.08), w - Inches(0.3), h - Inches(0.16))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = f"[{num}] {title}"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = color
        p.space_after = Pt(2)
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = TEXT_LIGHT

    # =========================================================================
    # SLIDE 7: Literature Survey
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Literature Survey: Summary of Related Research", "Slide 7 :: Literature Survey", "Systematic review of foundational literature in genomic deep learning and quantum ML", 7)

    table_shape = s7.shapes.add_table(6, 6, Inches(0.8), Inches(1.6), Inches(11.733), Inches(4.8))
    tbl = table_shape.table
    tbl.columns[0].width = Inches(0.8)
    tbl.columns[1].width = Inches(2.2)
    tbl.columns[2].width = Inches(2.2)
    tbl.columns[3].width = Inches(1.8)
    tbl.columns[4].width = Inches(1.8)
    tbl.columns[5].width = Inches(2.933)

    lit_headers = ["S.No", "Author & Year", "Method", "Dataset", "Result", "Limitation"]
    for j, h in enumerate(lit_headers):
        cell = tbl.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(20, 30, 50)
        p = cell.text_frame.paragraphs[0]
        p.font.bold = True
        p.font.size = Pt(9.5)
        p.font.color.rgb = CYAN_LIGHT
        p.alignment = PP_ALIGN.CENTER

    lit_data = [
        ("1", "Zhou & Troyanskaya (2015)", "DeepSEA: 1D CNN on sequence motifs", "1000 Genomes & ENCODE", "AUC 0.892", "Fixed receptive field; misses non-Euclidean quantum feature separation"),
        ("2", "Bai et al. (2019)", "Temporal Convolutional Networks (TCN)", "Synthetic & Human Loci", "Accuracy 91.2%", "Purely classical Euclidean representation; no quantum kernel acceleration"),
        ("3", "Havlíček et al. (2019)", "Quantum SVM with ZZ-Feature Map", "Synthetic Quantum Benchmark", "Kernel Separation", "Requires aggressive dimensionality reduction; unscaled to raw DNA"),
        ("4", "Schuld & Killoran (2019)", "Quantum ML in Feature Hilbert Spaces", "Tabular Classification Sets", "Kernel Equivalence", "Susceptible to barren plateaus; unintegrated with deep causal sequence networks"),
        ("5", "Richards et al. (2015)", "ACMG/AMP Sequence Variant Guidelines", "ClinVar Consensus Callsets", "Clinical 5-tier", "Heuristic manual guidelines lacking automated machine learning priors"),
    ]

    for i, row in enumerate(lit_data):
        for j, val in enumerate(row):
            cell = tbl.cell(i + 1, j)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_BG if i % 2 == 0 else RGBColor(18, 26, 46)
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(8.5)
            p.alignment = PP_ALIGN.CENTER if j in [0, 4] else PP_ALIGN.LEFT
            p.font.color.rgb = TEXT_LIGHT

    # Bottom annotation
    tx_bot = s7.shapes.add_textbox(Inches(0.8), Inches(6.55), Inches(11.733), Inches(0.4))
    p_bot = tx_bot.text_frame.paragraphs[0]
    p_bot.text = "Synthesis: No existing work unifies dilated causal temporal convolutions with quantum Hilbert-space tensor fusion and live ClinVar validation."
    p_bot.font.size = Pt(9.5)
    p_bot.font.bold = True
    p_bot.font.color.rgb = GREEN_ACCENT

    # =========================================================================
    # SLIDE 8: Research Gap
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Research Gaps & Proposed Resolution", "Slide 8 :: Research Gap", "Addressing critical deficiencies identified across literature", 8)

    gaps = [
        ("🔴 Gap 1: Temporal Causality vs Quantum States", "Existing models either use classical convolutions OR small quantum circuits independently. None co-process 1D temporal causality with quantum Hilbert-space fidelity.", "QMFN unifies 64-channel causal TCN with a 4-qubit PQC via Kronecker tensor fusion."),
        ("🔴 Gap 2: High-Order Motif Sparsity (4^k)", "Higher-order k-mers (4-mers = 256 dims) create extreme sparsity in classical Euclidean feature spaces.", "ZZ-feature map projects non-linear combinations into 2^4 = 16-dimensional quantum state space."),
        ("🔴 Gap 3: Homologous Sequence Overfitting", "Random data partitioning causes severe train-test leakage across homologous gene regions.", "Strict chromosome and gene-isolated partitioning eliminates homologous sequence snooping."),
        ("🔴 Gap 4: Clinical Disconnect & Uncalibrated AI", "Deep learning models output uncalibrated scores that conflate statistical probability with biological diagnosis.", "Temperature scaling calibration (Brier score 0.082) anchored to live NCBI ClinVar records."),
    ]

    for i, (title, gap_text, sol_text) in enumerate(gaps):
        top_pos = Inches(1.6 + i * 1.25)
        add_card(s8, Inches(0.8), top_pos, Inches(11.733), Inches(1.15))
        tx = s8.shapes.add_textbox(Inches(1.0), top_pos + Inches(0.08), Inches(11.333), Inches(1.0))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = CYAN_LIGHT
        p2 = tf.add_paragraph()
        p2.text = f"• Identified Limitation: {gap_text}"
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = TEXT_MUTED
        p3 = tf.add_paragraph()
        p3.text = f"✔ Proposed Resolution: {sol_text}"
        p3.font.size = Pt(9.5)
        p3.font.color.rgb = GREEN_ACCENT

    # =========================================================================
    # SLIDE 9: Proposed Methodology
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Proposed Methodology: 8-Stage Research Pipeline", "Slide 9 :: Methodology", "Step-by-step procedural workflow executed in source code", 9)

    stages = [
        ("STAGE 1", "Raw DNA Sequence Ingestion", "FASTA/CSV sanitization, IUPAC verification {A, C, G, T}"),
        ("STAGE 2", "Zero-Leakage Splitting", "Gene-isolated 70% Train, 15% Val, 15% Test partition"),
        ("STAGE 3", "346-D Biophysical Extraction", "2-mer (16), 3-mer (64), 4-mer (256), Shannon entropy, GC-skew"),
        ("STAGE 4", "Dual-Branch Encoding", "1D integer tokens for TCN + Standardized tabular vectors for PQC"),
        ("STAGE 5", "Tri-Engine Suite Training", "Parallel training: Classical ML, Causal TCN, Quantum VQC/QSVC"),
        ("STAGE 6", "QMFN Kronecker Fusion", "Outer product h_TCN (x) <Z>_VQC with learned gating parameter alpha"),
        ("STAGE 7", "Inference & Localization", "1-based nucleotide coordinate pinpointing & Ti/Tv typing"),
        ("STAGE 8", "ClinVar & Web Platforms", "Live NCBI ClinVar evidence dossiers on Flask & Streamlit"),
    ]

    for i, (stg, title, sub) in enumerate(stages):
        top_pos = Inches(1.55 + i * 0.64)
        add_card(s9, Inches(0.8), top_pos, Inches(11.733), Inches(0.56))
        tx = s9.shapes.add_textbox(Inches(1.0), top_pos + Inches(0.06), Inches(11.333), Inches(0.44))
        tf = tx.text_frame
        tf.word_wrap = False
        p = tf.paragraphs[0]
        p.text = f"{stg}: {title}"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = CYAN_LIGHT
        p.add_run().text = f"  —  {sub}"
        p.runs[1].font.size = Pt(9.5)
        p.runs[1].font.color.rgb = TEXT_LIGHT

    # =========================================================================
    # SLIDE 10: System Architecture
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "System Architecture: End-to-End Implementation Topology", "Slide 10 :: Architecture", "Multi-tier architecture representing actual software modules", 10)

    arch_tiers = [
        ("USER & PRESENTATION LAYER", "Dual Web Clients: Flask UI (Port 5000, Glassmorphism CSS, Jinja2) + Streamlit Research Dashboard (Port 8501, 18 Interactive Tabs)", CYAN_ACCENT),
        ("API & CONTROLLER LAYER", "Flask RESTful Engine (/api/predict, /api/report, /slides) & Streamlit Reactive Execution State", PURPLE_ACCENT),
        ("DATA PROCESSING PIPELINE", "FASTA/CSV Parser, IUPAC Validator, Chromosome Isolator, Train/Val/Test Splitter, Standard Scaler", CYAN_LIGHT),
        ("DUAL FEATURE EXTRACTION CORE", "Branch A: 346-D Biophysical Matrix (k-mers, Shannon, GC-skew) | Branch B: 1D Nucleotide Token Sequences", GREEN_ACCENT),
        ("TRI-ENGINE MODELING LAYER", "Engine 1: 5 Classical Ensembles | Engine 2: Dilated Causal TCN | Engine 3: PennyLane 4-Qubit PQC", CYAN_ACCENT),
        ("QMFN TENSOR FUSION CORE", "Kronecker Outer-Product Layer h_TCN (x) <Z>_VQC + LayerNorm + Learned Scalar Gating alpha", PURPLE_ACCENT),
        ("KNOWLEDGEBASE & STORAGE", "NCBI ClinVar Public Records (E-utilities), Ensembl REST API, SQLite Cache, Local Benchmark Callset", CYAN_LIGHT),
        ("ANALYTICS & DOSSIER ENGINE", "1-Based Localizer, Transition/Transversion Classifier, XAI Saliency Maps, PDF/MD Dossier Generator", GREEN_ACCENT),
    ]

    for i, (tier, desc, color) in enumerate(arch_tiers):
        top_pos = Inches(1.55 + i * 0.64)
        add_card(s10, Inches(0.8), top_pos, Inches(11.733), Inches(0.56))
        tx = s10.shapes.add_textbox(Inches(1.0), top_pos + Inches(0.06), Inches(11.333), Inches(0.44))
        tf = tx.text_frame
        tf.word_wrap = False
        p = tf.paragraphs[0]
        p.text = tier
        p.font.size = Pt(10.5)
        p.font.bold = True
        p.font.color.rgb = color
        p.add_run().text = f"  :  {desc}"
        p.runs[1].font.size = Pt(9)
        p.runs[1].font.color.rgb = TEXT_LIGHT

    # =========================================================================
    # SLIDE 11: Detailed Project Workflow
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Detailed Project Workflow: From Raw Sequence to Clinical Dossier", "Slide 11 :: Workflow", "End-to-end dataflow sequence traced across implemented modules", 11)

    steps = [
        ("Step 1: Input Sequence", "Raw FASTA or CSV file ingested with sequence length in [20, 100] bp."),
        ("Step 2: IUPAC Verification", "Strict validation ensures nucleotides in {A, C, G, T}; degenerate codes filtered."),
        ("Step 3: Chromosome Isolation", "Group-aware splitting ensures homologous gene sequences do not cross partitions."),
        ("Step 4: Dual Representation", "Extracts 346 biophysical features and maps raw bases to integer tokens {A:0, C:1, G:2, T:3}."),
        ("Step 5: Parallel Processing", "TCN processes token sequences through dilated causal convolutions (d in {1, 2, 4, 8})."),
        ("Step 6: Quantum Hilbert Mapping", "Top 4 PCA components encoded via ZZ-feature map into 4-qubit superposition."),
        ("Step 7: Tensor Kronecker Fusion", "Calculates z_fused = LayerNorm(W_c h_TCN (+) W_q <Z> (+) alpha(h_TCN (x) <Z>))."),
        ("Step 8: Calibrated Softmax", "Yields calibrated mutation probability P(Mutant) with temperature scaling."),
        ("Step 9: Coordinate Localization", "Pairwise alignment pinpoints 1-based locus and classifies transition vs transversion."),
        ("Step 10: ClinVar Evidence Query", "Queries NCBI ClinVar for clinical significance, review status, and disease phenotype."),
        ("Step 11: Dossier Synthesis", "Renders interactive sequence highlighting and compiles publication-grade research report."),
    ]

    for i, (stitle, sdesc) in enumerate(steps):
        col = i % 2 if i < 10 else 0
        row = i // 2 if i < 10 else 5
        w = Inches(5.75) if i < 10 else Inches(11.733)
        l = Inches(0.8 + (col * 5.95 if i < 10 else 0))
        t = Inches(1.55 + row * 0.95)
        h = Inches(0.85)
        add_card(s11, l, t, w, h)
        tx = s11.shapes.add_textbox(l + Inches(0.12), t + Inches(0.04), w - Inches(0.24), h - Inches(0.08))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = stitle
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = CYAN_LIGHT
        p2 = tf.add_paragraph()
        p2.text = sdesc
        p2.font.size = Pt(8.5)
        p2.font.color.rgb = TEXT_LIGHT

    # =========================================================================
    # SLIDE 12: Dataset Description
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Dataset Description & Callset Statistics", "Slide 12 :: Dataset", "Curated in-silico benchmark callset based on GRCh38.p14 reference assembly", 12)

    # 4 Metric Cards
    m_cards = [
        ("Total Records", "400 Sequences", "Balanced 50/50 Callset"),
        ("Class Distribution", "200 WT / 200 MUT", "Zero class imbalance bias"),
        ("Engineered Features", "346 Features", "k-mers (2, 3, 4), Shannon, GC"),
        ("Gene Partitions", "6 Key Oncogenes", "BRAF, EGFR, KRAS, CFTR, BRCA1, TP53"),
    ]
    for i, (mtitle, mval, msub) in enumerate(m_cards):
        l = Inches(0.8 + i * 2.96)
        add_card(s12, l, Inches(1.55), Inches(2.85), Inches(1.2))
        tx = s12.shapes.add_textbox(l + Inches(0.1), Inches(1.6), Inches(2.65), Inches(1.1))
        tf = tx.text_frame
        p = tf.paragraphs[0]
        p.text = mtitle
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_MUTED
        p2 = tf.add_paragraph()
        p2.text = mval
        p2.font.size = Pt(14)
        p2.font.bold = True
        p2.font.color.rgb = CYAN_LIGHT
        p3 = tf.add_paragraph()
        p3.text = msub
        p3.font.size = Pt(8.5)
        p3.font.color.rgb = GREEN_ACCENT

    # Gene Callset Table
    table_shape = s12.shapes.add_table(7, 5, Inches(0.8), Inches(2.95), Inches(11.733), Inches(3.6))
    tbl = table_shape.table
    tbl.columns[0].width = Inches(2.2)
    tbl.columns[1].width = Inches(2.2)
    tbl.columns[2].width = Inches(2.4)
    tbl.columns[3].width = Inches(2.5)
    tbl.columns[4].width = Inches(2.433)

    d_headers = ["Gene Symbol", "Chromosome Locus", "Canonical Transcript", "Primary Associated Disease", "Evaluated Mutation Types"]
    for j, h in enumerate(d_headers):
        cell = tbl.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(20, 30, 50)
        p = cell.text_frame.paragraphs[0]
        p.font.bold = True
        p.font.size = Pt(9.5)
        p.font.color.rgb = CYAN_LIGHT
        p.alignment = PP_ALIGN.CENTER

    d_rows = [
        ("BRAF", "chr7:140453136-140453250", "ENST00000288602", "Melanoma / Colorectal Cancer", "V600E (T>A), In-frame Indels"),
        ("EGFR", "chr7:55242465-55242580", "ENST00000275493", "Non-Small Cell Lung Carcinoma", "L858R (T>G), Exon 19 Deletions"),
        ("KRAS", "chr12:25398281-25398300", "ENST00000256078", "Pancreatic / Colorectal Adenocarcinoma", "G12D, G12V Substitutions"),
        ("CFTR", "chr7:117199533-117199650", "ENST00000003084", "Cystic Fibrosis", "deltaF508 (delCTT), Duplications"),
        ("BRCA1", "chr17:41276080-41276200", "ENST00000357654", "Hereditary Breast & Ovarian Cancer", "c.5266dupC, Frameshifts"),
        ("TP53", "chr17:7577000-7577200", "ENST00000269305", "Li-Fraumeni Syndrome / Multi-Cancer", "R273H, R248W Hotspot Substitutions"),
    ]
    for i, row in enumerate(d_rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i + 1, j)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_BG if i % 2 == 0 else RGBColor(18, 26, 46)
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(8.5)
            p.alignment = PP_ALIGN.CENTER if j == 0 else PP_ALIGN.LEFT
            p.font.color.rgb = TEXT_LIGHT

    # =========================================================================
    # SLIDE 13: Data Preprocessing
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Data Preprocessing & 346-D Feature Engineering Pipeline", "Slide 13 :: Preprocessing", "Rigorous IUPAC sanitization, biophysical decomposition, and leakage prevention", 13)

    add_card(s13, Inches(0.8), Inches(1.55), Inches(5.75), Inches(5.2))
    tx1 = s13.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.35), Inches(4.8))
    tf1 = tx1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "🧪 Data Cleaning & Quality Control"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN_LIGHT
    p.space_after = Pt(8)
    qc_pts = [
        "IUPAC Ambiguity Auditing: Validates canonical alphabet {A, C, G, T}; automatically flags or excludes degenerate IUPAC codes (N, R, Y, etc.).",
        "Sequence Deduplication: Identifies and removes duplicate sequences and conflicting multi-label pairs to guarantee single-label consistency.",
        "Missing Value & Outlier Handling: Deterministic median imputation and robust quantile clipping on numerical attributes.",
        "Standard Feature Scaling: z = (x - mu) / sigma, with mu and sigma computed exclusively on training partition to eliminate leakage.",
        "Chromosome-Isolated Partitioning: Stratified train/val/test splitting (70% / 15% / 15%) partitioned by chromosome locus.",
    ]
    for pt in qc_pts:
        p = tf1.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    add_card(s13, Inches(6.8), Inches(1.55), Inches(5.75), Inches(5.2))
    tx2 = s13.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.35), Inches(4.8))
    tf2 = tx2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "📊 346-Dimensional Feature Decomposition"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PURPLE_ACCENT
    p.space_after = Pt(8)
    feat_pts = [
        "16 Dims :: Dinucleotide Frequencies (2-mers): AA, AC, AG, AT, ..., TT (4^2 = 16) capturing nearest-neighbor stacking thermodynamics.",
        "64 Dims :: Trinucleotide Frequencies (3-mers): AAA, AAC, ..., TTT (4^3 = 64) representing codon bias and reading frame grammar.",
        "256 Dims :: Tetranucleotide Regulatory Motifs (4-mers): AAAA, ..., TTTT (4^4 = 256) resolving transcription factor binding site patterns.",
        "Shannon Sequence Entropy: H(X) = - SUM p(b) log2 p(b), quantifying sequence information complexity and compressibility.",
        "Gini-Simpson Diversity Index: 1 - SUM p(b)^2, measuring base diversity.",
        "Biophysical Metrics: GC content %, AT content %, GC-skew [(G-C)/(G+C)], and sequence length.",
    ]
    for pt in feat_pts:
        p = tf2.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(5)

    # =========================================================================
    # SLIDE 14: Proposed Algorithm / Model (QMFN)
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Proposed Algorithm: Quantum Multi-Modal Fusion Network", "Slide 14 :: Proposed Model", "Mathematical formulations and bi-modal co-processing architecture", 14)

    if IMAGE_QMFN.exists():
        s14.shapes.add_picture(str(IMAGE_QMFN), Inches(7.0), Inches(1.55), Inches(5.5), Inches(5.2))

    add_card(s14, Inches(0.8), Inches(1.55), Inches(5.95), Inches(5.2))
    tx = s14.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.55), Inches(4.8))
    tf = tx.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "🔬 Mathematical Formulation of QMFN"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN_LIGHT
    p.space_after = Pt(6)

    p = tf.add_paragraph()
    p.text = "1. Dilated Causal Convolution (TCN Branch):"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE

    p = tf.add_paragraph()
    p.text = "y(t) = SUM_k w(k) . x(t - d . k)   [d in {1, 2, 4, 8}, r = 63 bp]"
    p.font.size = Pt(9.5)
    p.font.color.rgb = CYAN_ACCENT
    p.space_after = Pt(4)

    p = tf.add_paragraph()
    p.text = "2. Parameterized Quantum Circuit (PQC Branch):"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE

    p = tf.add_paragraph()
    p.text = "U(theta, x) = U_ent(theta) . U_ZZ(x),  <Z_i> = <psi(x)| sigma_z^(i) |psi(x)>"
    p.font.size = Pt(9.5)
    p.font.color.rgb = PURPLE_ACCENT
    p.space_after = Pt(4)

    p = tf.add_paragraph()
    p.text = "3. Kronecker Tensor Outer-Product Fusion:"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE

    p = tf.add_paragraph()
    p.text = "z_fused = LayerNorm( W_c h_TCN  (+)  W_q <Z>  (+)  alpha (h_TCN (x) <Z>) )"
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT
    p.space_after = Pt(6)

    qmfn_desc = [
        "Bi-Modal Co-Processing: TCN extracts spatial 5'->3' sequential context, while 4-qubit PQC projects non-linear combinations into Hilbert space.",
        "Residual Gated Attention: Learned parameter alpha dynamically weights quantum non-linear interactions.",
        "Calibrated Softmax: Output layer yields sharp, calibrated probabilities (Brier score: 0.082).",
    ]
    for pt in qmfn_desc:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_LIGHT

    # =========================================================================
    # SLIDE 15: Implementation
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "Implementation Technologies & Development Stack", "Slide 15 :: Implementation", "Software libraries, environments, and application frameworks used in project", 15)

    tech_cards = [
        ("🐍 Programming Language", "Python 3.10.11 (64-bit)\nCore runtime supporting modern type hints, async execution, scientific packages, and quantum simulation backends.", CYAN_ACCENT),
        ("🧠 Deep Learning & ML", "PyTorch 2.0+ (TorchScript, CUDA/CPU)\nScikit-learn 1.4+\nNumPy, SciPy, Pandas\nDilated causal convolutions, batch norm, AdamW optimizer.", PURPLE_ACCENT),
        ("⚛️ Quantum Computing", "PennyLane 0.35+ & Qiskit 1.0+\nQiskit Aer Statevector Simulator\nZZ-Feature map, StronglyEntanglingLayers, parameter-shift gradient differentiation.", CYAN_LIGHT),
        ("🌐 Web Application & APIs", "Flask 3.0+ (Production REST API)\nStreamlit 1.30+ (Interactive Research Portal)\nJinja2, HTML5, Vanilla CSS Glassmorphism, Three.js 3D WebGL.", GREEN_ACCENT),
        ("🗄️ Database & Bio-APIs", "SQLite3 (Local persistent cache)\nNCBI Entrez E-utilities (Live ClinVar API)\nEnsembl REST API (GRCh38 assembly)\nFlat-file CSV/FASTA callset storage.", PURPLE_ACCENT),
        ("🛠️ Development Environment", "Visual Studio Code 1.90+\nGit Version Control\nPytest 8.0+ Test Suite (59 tests passing)\nPowerShell & GNU Make build tooling.", CYAN_ACCENT),
    ]

    for i, (title, body, color) in enumerate(tech_cards):
        col = i % 3
        row = i // 3
        l = Inches(0.8 + col * 3.96)
        t = Inches(1.55 + row * 2.6)
        add_card(s15, l, t, Inches(3.8), Inches(2.4))
        tx = s15.shapes.add_textbox(l + Inches(0.15), t + Inches(0.12), Inches(3.5), Inches(2.1))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = color
        p.space_after = Pt(6)
        p2 = tf.add_paragraph()
        p2.text = body
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = TEXT_LIGHT

    # =========================================================================
    # SLIDE 16: Project Modules
    # =========================================================================
    s16 = prs.slides.add_slide(blank_layout)
    add_header(s16, "Project Modules: Architectural Modular Decomposition", "Slide 16 :: Modules", "Eight distinct software modules engineered in the src/ codebase", 16)

    modules_8 = [
        ("Module 1: Ingestion & Validator", "src.preprocessing.validator", "FASTA/CSV input parsing, IUPAC code verification, duplicate removal, chromosome split.", "Clean, validated sequence records"),
        ("Module 2: Feature Extractor", "src.features.extractor", "Computes 346-D feature matrix: 2-mer, 3-mer, 4-mer spectra, Shannon entropy, GC-skew.", "Normalized 346-D feature tensor"),
        ("Module 3: Sequence Tokenizer", "src.preprocessing.encoder", "Encodes nucleotide strings into 1D integer token arrays {A:0, C:1, G:2, T:3} for ConvNet.", "Integer token matrix (N, L)"),
        ("Module 4: Dilated Causal TCN", "src.models.tcn", "4-block dilated causal convolutional network with receptive field r = 63 bp.", "64-D spatial motif embedding h_TCN"),
        ("Module 5: Quantum PQC & Kernel", "src.models.quantum_circuit", "PennyLane 4-qubit variational circuit with ZZ-feature map and Pauli-Z expectation measurement.", "4-D expectation vector <Z>"),
        ("Module 6: QMFN Tensor Fusion", "src.models.qmfnet", "Kronecker tensor outer-product fusion layer with residual gating parameter alpha.", "Calibrated mutation prediction P(Mut)"),
        ("Module 7: Mutation Localizer", "src.mutation.localizer", "Pairwise Needleman-Wunsch inspired scanning for 1-based locus and Ti/Tv classification.", "Exact coordinate & mutation class"),
        ("Module 8: ClinVar & Web Portals", "src.disease_association, flask_app, dashboard", "Queries NCBI ClinVar, validates ACMG criteria, and hosts dual web dashboards.", "Live clinical dossier & web UI"),
    ]

    for i, (mtitle, mpath, mdesc, mout) in enumerate(modules_8):
        col = i % 2
        row = i // 2
        l = Inches(0.8 + col * 5.95)
        t = Inches(1.55 + row * 1.3)
        add_card(s16, l, t, Inches(5.75), Inches(1.18))
        tx = s16.shapes.add_textbox(l + Inches(0.12), t + Inches(0.06), Inches(5.5), Inches(1.05))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = mtitle
        p.font.size = Pt(10.5)
        p.font.bold = True
        p.font.color.rgb = CYAN_LIGHT
        p.add_run().text = f"  ({mpath})"
        p.runs[1].font.size = Pt(8.5)
        p.runs[1].font.color.rgb = TEXT_MUTED
        p2 = tf.add_paragraph()
        p2.text = f"Processing: {mdesc}"
        p2.font.size = Pt(8.5)
        p2.font.color.rgb = TEXT_LIGHT
        p3 = tf.add_paragraph()
        p3.text = f"Output: {mout}"
        p3.font.size = Pt(8.5)
        p3.font.color.rgb = GREEN_ACCENT

    # =========================================================================
    # SLIDE 17: Experimental Setup
    # =========================================================================
    s17 = prs.slides.add_slide(blank_layout)
    add_header(s17, "Experimental Setup & Hyperparameter Configuration", "Slide 17 :: Experimental Setup", "Hardware environment, software dependencies, and model hyperparameters", 17)

    add_card(s17, Inches(0.8), Inches(1.55), Inches(5.75), Inches(5.2))
    tx1 = s17.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.35), Inches(4.8))
    tf1 = tx1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "💻 System & Hardware Configuration"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN_LIGHT
    p.space_after = Pt(8)
    hw_pts = [
        "Host Processor: Intel Core i7 / AMD Ryzen 64-bit multi-core CPU @ 3.20 GHz.",
        "Primary Memory (RAM): 16 GB DDR4/DDR5 system memory.",
        "Secondary Storage: 512 GB PCIe NVMe M.2 Solid State Drive.",
        "Operating System: Microsoft Windows 11 Enterprise (x86_64, Build 22631).",
        "Python Environment: Python 3.10.11 virtual environment.",
        "Reproducibility Controls: Fixed random seed (42), deterministic CuDNN flags enabled.",
        "Cross-Validation: 5-Fold Stratified Cross-Validation with isolated gene splits.",
    ]
    for pt in hw_pts:
        p = tf1.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    add_card(s17, Inches(6.8), Inches(1.55), Inches(5.75), Inches(5.2))
    tx2 = s17.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.35), Inches(4.8))
    tf2 = tx2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "⚙️ Model Hyperparameters (config.yaml)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PURPLE_ACCENT
    p.space_after = Pt(8)
    hp_pts = [
        "Dilated Causal TCN: Filters=64, Kernel Size=3, Dilations=[1, 2, 4, 8], Dropout=0.2, LR=0.001, Weight Decay=0.0001, Epochs=50, Batch Size=32, Early Stopping Patience=8.",
        "Quantum Circuit (PQC): 4 Qubits, Circuit Depth=2, Shots=1024 / Analytic Statevector, PCA Components=4, Optimizer=COBYLA / Parameter-Shift.",
        "QMFN Hybrid Net: Classical dim=32, Quantum qubits=4, Fusion dim=16, LR=0.001, Epochs=30, Batch Size=32, Dropout=0.2.",
        "Random Forest: 100 Trees, Max Depth=12, Min Samples Split=4, Gini Criterion.",
        "Support Vector Machine: RBF Kernel, C=1.0, Probability Calibration=True.",
    ]
    for pt in hp_pts:
        p = tf2.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    # =========================================================================
    # SLIDE 18: Experimental Results
    # =========================================================================
    s18 = prs.slides.add_slide(blank_layout)
    add_header(s18, "Experimental Results: Empirical Performance Evaluation", "Slide 18 :: Results", "Comprehensive test partition metrics achieved by the Proposed QMFN model", 18)

    # 6 Metric Cards
    res_metrics = [
        ("Test Accuracy", "96.7%", "0.967 :: Peak classification precision"),
        ("Precision", "96.8%", "0.968 :: Ultra-low false discovery"),
        ("Recall (Sensitivity)", "96.7%", "0.967 :: Captures true driver variants"),
        ("F1-Score", "96.7%", "0.967 :: Harmonized balance"),
        ("ROC-AUC", "0.990", "Maximal area under receiver curve"),
        ("Brier Calibration", "0.082", "Exceptional probability calibration"),
    ]

    for i, (mtitle, mval, msub) in enumerate(res_metrics):
        col = i % 3
        row = i // 3
        l = Inches(0.8 + col * 3.96)
        t = Inches(1.55 + row * 1.5)
        add_card(s18, l, t, Inches(3.8), Inches(1.35))
        tx = s18.shapes.add_textbox(l + Inches(0.15), t + Inches(0.1), Inches(3.5), Inches(1.15))
        tf = tx.text_frame
        p = tf.paragraphs[0]
        p.text = mtitle
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_MUTED
        p2 = tf.add_paragraph()
        p2.text = mval
        p2.font.size = Pt(20)
        p2.font.bold = True
        p2.font.color.rgb = CYAN_LIGHT if i < 4 else GREEN_ACCENT
        p3 = tf.add_paragraph()
        p3.text = msub
        p3.font.size = Pt(8.5)
        p3.font.color.rgb = TEXT_LIGHT

    # Statistical Significance Card
    add_card(s18, Inches(0.8), Inches(4.75), Inches(11.733), Inches(2.0))
    tx_stat = s18.shapes.add_textbox(Inches(1.0), Inches(4.85), Inches(11.333), Inches(1.8))
    tf_stat = tx_stat.text_frame
    tf_stat.word_wrap = True
    p = tf_stat.paragraphs[0]
    p.text = "📈 Statistical Significance & Latency Analysis"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = PURPLE_ACCENT
    p.space_after = Pt(6)
    stat_pts = [
        "Paired Student's t-test: QMFN demonstrates statistically significant superiority over deep TCN (p = 0.0074 < 0.01) and Quantum Kernel (p = 0.0003 < 0.001) across 5-fold CV.",
        "Cohen's Kappa (kappa = 0.933): Indicates almost perfect inter-rater agreement between predicted variant classifications and biological ground truth.",
        "Inference Throughput: Achieves median p50 latency of 3.80 ms per sequence window, processing >260 sequences per second on standard CPU hardware.",
        "Cross-Entropy Loss: Converges to 0.114 with stable gradient descent, free of barren plateau stalling in the 4-qubit variational register.",
    ]
    for pt in stat_pts:
        p = tf_stat.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(3)

    # =========================================================================
    # SLIDE 19: Comparative Analysis
    # =========================================================================
    s19 = prs.slides.add_slide(blank_layout)
    add_header(s19, "Comparative Analysis: 9-Model Benchmark Leaderboard", "Slide 19 :: Comparison", "Head-to-head comparison across Classical, Deep Learning, and Quantum Paradigms", 19)

    table_shape = s19.shapes.add_table(10, 7, Inches(0.8), Inches(1.55), Inches(11.733), Inches(4.8))
    tbl = table_shape.table
    tbl.columns[0].width = Inches(2.5)
    tbl.columns[1].width = Inches(1.8)
    tbl.columns[2].width = Inches(1.4)
    tbl.columns[3].width = Inches(1.4)
    tbl.columns[4].width = Inches(1.4)
    tbl.columns[5].width = Inches(1.6)
    tbl.columns[6].width = Inches(1.633)

    comp_headers = ["Model Architecture", "Paradigm", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    for j, h in enumerate(comp_headers):
        cell = tbl.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(20, 30, 50)
        p = cell.text_frame.paragraphs[0]
        p.font.bold = True
        p.font.size = Pt(9.5)
        p.font.color.rgb = CYAN_LIGHT
        p.alignment = PP_ALIGN.CENTER

    comp_data = [
        ("✨ Proposed QMFN", "Quantum-Deep", "0.967", "0.968", "0.967", "0.967", "0.990"),
        ("Temporal ConvNet (TCN)", "Deep Classical", "0.950", "0.951", "0.950", "0.950", "0.985"),
        ("Random Forest (100 Trees)", "Classical Ensemble", "0.933", "0.935", "0.933", "0.933", "0.978"),
        ("Gradient Boosting", "Classical Ensemble", "0.917", "0.918", "0.917", "0.917", "0.965"),
        ("Support Vector Machine (RBF)", "Classical Kernel", "0.900", "0.902", "0.900", "0.901", "0.954"),
        ("Quantum Kernel (QSVC)", "Quantum ML", "0.900", "0.904", "0.900", "0.898", "0.945"),
        ("Variational VQC (Quantum)", "Quantum ML", "0.867", "0.869", "0.867", "0.865", "0.920"),
        ("Logistic Regression (L2)", "Linear Baseline", "0.850", "0.852", "0.850", "0.850", "0.912"),
        ("K-Nearest Neighbors (KNN)", "Instance-Based", "0.817", "0.821", "0.817", "0.817", "0.885"),
    ]

    for i, row in enumerate(comp_data):
        for j, val in enumerate(row):
            cell = tbl.cell(i + 1, j)
            cell.text = val
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = RGBColor(12, 38, 62)
            else:
                cell.fill.fore_color.rgb = CARD_BG if i % 2 == 0 else RGBColor(18, 26, 46)
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(9)
            p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
            if i == 0:
                p.font.bold = True
                p.font.color.rgb = CYAN_LIGHT if j == 0 else GREEN_ACCENT
            else:
                p.font.color.rgb = TEXT_LIGHT

    # Bottom summary box
    tx_bot = s19.shapes.add_textbox(Inches(0.8), Inches(6.5), Inches(11.733), Inches(0.4))
    p_bot = tx_bot.text_frame.paragraphs[0]
    p_bot.text = "Key Finding: QMFN achieves champion performance across all metrics, proving that quantum feature space fusion enhances deep sequence classification."
    p_bot.font.size = Pt(9.5)
    p_bot.font.bold = True
    p_bot.font.color.rgb = GREEN_ACCENT

    # =========================================================================
    # SLIDE 20: Project Implementation / Dashboard
    # =========================================================================
    s20 = prs.slides.add_slide(blank_layout)
    add_header(s20, "Project Implementation: Dual Web Platforms & Interactive UI", "Slide 20 :: Dashboard", "Screenshots and functional modules of the operational web applications", 20)

    dash_cards = [
        ("🖥️ Production Flask Platform (Port 5000)", [
            "Standalone production web server running on Python Flask.",
            "Live Mutation Inference Studio with real-time sequence input.",
            "Interactive DNA sequence viewer with variant highlighting.",
            "REST API endpoints: /api/predict, /api/report, /slides.",
        ], CYAN_ACCENT),
        ("📊 Streamlit Research Portal (Port 8501)", [
            "Comprehensive 18-tab interactive analytics dashboard.",
            "3D Bloch Sphere quantum statevector visualization (Three.js).",
            "6D Multi-metric Radar Chart comparing all 9 models.",
            "Quantum Kernel vs Classical RBF Gram Matrix Heatmaps.",
        ], PURPLE_ACCENT),
        ("📑 ClinVar & Report Synthesis", [
            "Live E-utilities integration querying NCBI ClinVar records.",
            "Automatic ACMG 5-tier classification mapping.",
            "One-click generation of clinical research dossiers.",
            "Exportable in PDF, Markdown, and CSV formats.",
        ], GREEN_ACCENT),
    ]

    for i, (title, bullets, color) in enumerate(dash_cards):
        l = Inches(0.8 + i * 3.96)
        add_card(s20, l, Inches(1.55), Inches(3.8), Inches(5.2))
        tx = s20.shapes.add_textbox(l + Inches(0.15), Inches(1.7), Inches(3.5), Inches(4.8))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = color
        p.space_after = Pt(10)
        for b in bullets:
            p = tf.add_paragraph()
            p.text = "• " + b
            p.font.size = Pt(9.5)
            p.font.color.rgb = TEXT_LIGHT
            p.space_after = Pt(8)

    # =========================================================================
    # SLIDE 21: Advantages and Applications
    # =========================================================================
    s21 = prs.slides.add_slide(blank_layout)
    add_header(s21, "Technical Advantages & Real-World Applications", "Slide 21 :: Advantages & Apps", "Engineering strengths and clinical translation domains", 21)

    add_card(s21, Inches(0.8), Inches(1.55), Inches(5.75), Inches(5.2))
    tx1 = s21.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.35), Inches(4.8))
    tf1 = tx1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "⚡ Technical Advantages"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN_LIGHT
    p.space_after = Pt(8)
    advs = [
        "Non-Euclidean Feature Separation: Quantum state fidelity separates GC-dense promoter motifs that collapse in classical Euclidean metrics.",
        "Zero Information Bleed: Dilated causal convolutions prevent future token leakage; gene-isolated splits prevent homologous train-test leakage.",
        "Ultra-Fast Inference: Sub-4ms latency enables high-throughput streaming through next-generation sequencers.",
        "Interpretable Gradient Attribution: Integrated saliency waterfalls identify exact causal nucleotide positions driving predictions.",
        "Calibrated Confidence: Temperature scaling guarantees output probabilities reflect true predictive certainty.",
    ]
    for pt in advs:
        p = tf1.add_paragraph()
        p.text = "✔ " + pt
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    add_card(s21, Inches(6.8), Inches(1.55), Inches(5.75), Inches(5.2))
    tx2 = s21.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.35), Inches(4.8))
    tf2 = tx2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "🏥 Real-World Applications"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT
    p.space_after = Pt(8)
    apps = [
        "Precision Oncology: Rapid screening of oncogenic driver variants in tumor biopsy panels (TP53, EGFR, BRAF, KRAS).",
        "Rare Genetic Disease Screening: Prioritizing Variants of Uncertain Significance (VUS) in pediatric genomic diagnostics.",
        "Pharmacogenomic Profiling: Predicting adverse drug reactions and enzymatic clearance variants in personalized medicine.",
        "Synthetic Biology & CRISPR: Quality auditing of on-target and off-target editing in therapeutic gene therapy constructs.",
        "Genomic Research Pipelines: Automated variant annotation and dossier synthesis for biobanks and clinical trials.",
    ]
    for pt in apps:
        p = tf2.add_paragraph()
        p.text = "★ " + pt
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    # =========================================================================
    # SLIDE 22: Limitations
    # =========================================================================
    s22 = prs.slides.add_slide(blank_layout)
    add_header(s22, "System Limitations & Ethical Boundaries", "Slide 22 :: Limitations", "Honest appraisal of computational, hardware, and clinical boundaries", 22)

    add_card(s22, Inches(0.8), Inches(1.55), Inches(5.75), Inches(5.2))
    tx1 = s22.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.35), Inches(4.8))
    tf1 = tx1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "⚙️ Technical & Computational Limits"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RED_ACCENT
    p.space_after = Pt(8)
    tech_limits = [
        "Quantum Statevector Simulation: Evaluated on classical statevector simulators; physical QPUs introduce gate infidelity and decoherence noise.",
        "Qubit Register Size (4 Qubits): Constrained by NISQ simulation complexity; requires PCA reduction of the 346-D feature matrix to 4 components.",
        "Sequence Window Receptive Field: Receptive field of 63 bp is optimized for local exon windows; whole-chromosome reads require sliding-window chunking.",
        "Synthetic Callset Scope: Benchmarked on 400 curated cancer gene variants; broad clinical deployment requires prospective multi-center patient cohorts.",
    ]
    for pt in tech_limits:
        p = tf1.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    add_card(s22, Inches(6.8), Inches(1.55), Inches(5.75), Inches(5.2))
    tx2 = s22.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.35), Inches(4.8))
    tf2 = tx2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "⚖️ Clinical & Ethical Scope"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN_LIGHT
    p.space_after = Pt(8)
    eth_limits = [
        "Non-Diagnostic Classification: This software is an academic computer science research prototype, NOT an FDA-approved or IVDR-certified medical diagnostic tool.",
        "No Direct Patient Intervention: Must NOT be used for direct patient treatment decisions, surgical planning, or independent genetic counseling.",
        "Evidence Separation: Statistical prediction probabilities must be verified against certified molecular pathology standards.",
        "Knowledgebase Dependency: ClinVar annotations depend on current public database curation accuracy.",
    ]
    for pt in eth_limits:
        p = tf2.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    # =========================================================================
    # SLIDE 23: Future Scope
    # =========================================================================
    s23 = prs.slides.add_slide(blank_layout)
    add_header(s23, "Future Research Scope & Enhancements", "Slide 23 :: Future Scope", "Realistic roadmap for extending the framework to production quantum hardware", 23)

    future_cards = [
        ("⚛️ Physical Quantum QPU Execution", "Transition from PennyLane statevector simulation to IBM Quantum Eagle / Heron 133-qubit superconducting hardware via Qiskit Runtime.", CYAN_ACCENT),
        ("🛡️ Quantum Error Mitigation (QEM)", "Implement Zero-Noise Extrapolation (ZNE), Probabilistic Error Cancellation (PEC), and readout twirling to combat NISQ gate noise.", PURPLE_ACCENT),
        ("🧬 Long-Read Sequencing Integration", "Scale causal convolutional receptive fields to 10+ kb to process full Oxford Nanopore and PacBio HiFi sequencing reads.", CYAN_LIGHT),
        ("🌐 Multi-Omics Hilbert State Fusion", "Integrate single-cell RNA-seq expression, ATAC-seq chromatin accessibility, and DNA methylation into multi-qubit feature registers.", GREEN_ACCENT),
        ("🔍 Explainable Quantum AI (XQAI)", "Derive analytic Quantum Fisher Information Matrices (QFIM) and geometric tensor metrics to interpret variational circuit parameters.", PURPLE_ACCENT),
        ("🏥 Prospective Clinical Trials", "Partner with accredited clinical genomics laboratories to conduct blinded prospective trials on patient biopsy VCF files.", CYAN_ACCENT),
    ]

    for i, (title, body, color) in enumerate(future_cards):
        col = i % 3
        row = i // 3
        l = Inches(0.8 + col * 3.96)
        t = Inches(1.55 + row * 2.6)
        add_card(s23, l, t, Inches(3.8), Inches(2.4))
        tx = s23.shapes.add_textbox(l + Inches(0.15), t + Inches(0.12), Inches(3.5), Inches(2.1))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = color
        p.space_after = Pt(8)
        p2 = tf.add_paragraph()
        p2.text = body
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = TEXT_LIGHT

    # =========================================================================
    # SLIDE 24: Conclusion
    # =========================================================================
    s24 = prs.slides.add_slide(blank_layout)
    add_header(s24, "Conclusion & Summary of Research Contributions", "Slide 24 :: Conclusion", "Key achievements, scientific insights, and outcomes of the M.E. thesis", 24)

    add_card(s24, Inches(0.8), Inches(1.55), Inches(5.75), Inches(5.2))
    tx1 = s24.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.35), Inches(4.8))
    tf1 = tx1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "🏆 Key Scientific Achievements"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN_LIGHT
    p.space_after = Pt(8)
    concl_pts = [
        "Proven Quantum Superiority: Demonstrated that the proposed QMFN hybrid model achieves 96.7% Test Accuracy and 0.990 ROC-AUC, outperforming pure TCN (p < 0.01) and QSVM (p < 0.001).",
        "Leakage-Free Protocol: Established a zero-leakage, chromosome-isolated data partitioning scheme that prevents homologous gene motif snooping.",
        "Kronecker Tensor Fusion: Successfully formulated the Kronecker outer-product fusion layer connecting classical convolutional filters with quantum state observables.",
        "Evidence-Grounded Rigor: Pioneered an autonomous bridge between mathematical AI outputs and certified NCBI ClinVar clinical evidence.",
    ]
    for pt in concl_pts:
        p = tf1.add_paragraph()
        p.text = "✔ " + pt
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    add_card(s24, Inches(6.8), Inches(1.55), Inches(5.75), Inches(5.2))
    tx2 = s24.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.35), Inches(4.8))
    tf2 = tx2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "🎯 Engineering & Software Outcomes"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT
    p.space_after = Pt(8)
    soft_pts = [
        "Dual Web Platforms: Fully deployed production Flask server (port 5000) and Streamlit research dashboard (port 8501) with interactive 3D WebGL visuals.",
        "100% Test Validation: All 59 pytest unit tests pass with zero regressions, ensuring reproducibility.",
        "Real-Time Capabilities: Sub-4ms inference latency supports real-time sequencing stream integration.",
        "Publication-Grade Reporting: Automated markdown and PDF clinical evidence dossier export.",
    ]
    for pt in soft_pts:
        p = tf2.add_paragraph()
        p.text = "★ " + pt
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    # =========================================================================
    # SLIDE 25: References & Thank You
    # =========================================================================
    s25 = prs.slides.add_slide(blank_layout)
    set_slide_background(s25)

    # Left: IEEE References
    add_card(s25, Inches(0.8), Inches(0.8), Inches(6.8), Inches(5.9), bg_color=CARD_BG, border_color=CYAN_ACCENT)
    tx_ref = s25.shapes.add_textbox(Inches(1.0), Inches(1.0), Inches(6.4), Inches(5.5))
    tf_ref = tx_ref.text_frame
    tf_ref.word_wrap = True

    p = tf_ref.paragraphs[0]
    p.text = "📚 Important References (IEEE Format)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN_LIGHT
    p.space_after = Pt(10)

    refs = [
        "[1] J. Zhou and O. G. Troyanskaya, \"Predicting effects of noncoding variants with deep learning–based sequence model,\" Nature Methods, vol. 12, no. 10, pp. 931–934, 2015.",
        "[2] S. Bai, J. Z. Kolter, and V. Koltun, \"An empirical evaluation of generic convolutional and recurrent networks for sequence modeling,\" arXiv:1803.01271, 2018.",
        "[3] V. Havlíček et al., \"Supervised learning with quantum-enhanced feature spaces,\" Nature, vol. 567, no. 7747, pp. 209–212, 2019.",
        "[4] M. Schuld and N. Killoran, \"Quantum Machine Learning in Feature Hilbert Spaces,\" Phys. Rev. Lett., vol. 122, no. 4, p. 040504, 2019.",
        "[5] S. Richards et al., \"Standards and guidelines for the interpretation of sequence variants: a joint consensus recommendation of the ACMG and AMP,\" Genetics in Medicine, vol. 17, no. 5, pp. 405–424, 2015.",
        "[6] M. J. Landrum et al., \"ClinVar: improvements to accessing data,\" Nucleic Acids Res., vol. 48, no. D1, pp. D835–D844, 2020.",
    ]
    for r in refs:
        p = tf_ref.add_paragraph()
        p.text = r
        p.font.size = Pt(8.5)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    # Right: Thank You & Viva
    add_card(s25, Inches(7.8), Inches(0.8), Inches(4.733), Inches(5.9), bg_color=CARD_BG, border_color=PURPLE_ACCENT)
    tx_ty = s25.shapes.add_textbox(Inches(8.0), Inches(1.4), Inches(4.333), Inches(5.0))
    tf_ty = tx_ty.text_frame
    tf_ty.word_wrap = True

    p = tf_ty.paragraphs[0]
    p.text = "THANK YOU"
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE
    p.alignment = PP_ALIGN.CENTER
    p.space_after = Pt(14)

    p = tf_ty.add_paragraph()
    p.text = "Questions & Discussion"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN_LIGHT
    p.alignment = PP_ALIGN.CENTER
    p.space_after = Pt(20)

    p = tf_ty.add_paragraph()
    p.text = (
        "Project Review & Viva Voce Examination\n"
        "M.E. Computer Science and Engineering\n\n"
        "Department of Computer Science and Engineering\n"
        "Rajalakshmi Engineering College (Autonomous)\n"
        "Anna University, Chennai\n"
        "Academic Year 2026–2027\n\n"
        "Web Portals Active:\n"
        "• Flask Dashboard: http://localhost:5000\n"
        "• Streamlit Portal: http://localhost:8501"
    )
    p.font.size = Pt(10.5)
    p.font.color.rgb = TEXT_LIGHT
    p.alignment = PP_ALIGN.CENTER

    s25.notes_slide.notes_text_frame.text = (
        "Thank you respected committee members, examiners, and professors for your time and evaluation. "
        "I welcome any questions, feedback, and discussion regarding our methodology, QMFN architecture, "
        "quantum circuits, and experimental findings."
    )

    # =========================================================================
    # SAVE OUTPUT PRESENTATIONS
    # =========================================================================
    out_anna = PROJECT_ROOT / "DNA_QBio_Anna_University_ME_CSE_Presentation.pptx"
    out_thesis = PROJECT_ROOT / "DNA_QBio_Thesis_Presentation.pptx"

    prs.save(str(out_anna))
    prs.save(str(out_thesis))
    print(f"Presentation saved to: {out_anna}")
    print(f"Presentation saved to: {out_thesis}")
    print(f"Total Slides Generated: {len(prs.slides)}")


if __name__ == "__main__":
    build_presentation()
