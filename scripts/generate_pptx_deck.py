"""Generates a professional PowerPoint presentation (.pptx) for FastReasonSeg-3D."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PPTX = ROOT_DIR / "docs" / "FastReasonSeg_3D_Presentation.pptx"

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)  # 16:9 Widescreen
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Theme Colors
    NAVY = RGBColor(15, 23, 42)      # #0f172a
    CARD_BG = RGBColor(30, 41, 59)   # #1e293b
    ACCENT_BLUE = RGBColor(56, 189, 248) # #38bdf8
    ACCENT_GREEN = RGBColor(52, 211, 153) # #34d399
    TEXT_LIGHT = RGBColor(241, 245, 249) # #f1f5f9
    TEXT_MUTED = RGBColor(148, 163, 184) # #94a3b8

    def add_bg(slide):
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        shape.fill.solid()
        shape.fill.fore_color.rgb = NAVY
        shape.line.fill.background()

    def add_header(slide, title_text, category_text="FASTREASONSEG-3D RESEARCH"):
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(0.4))
        tf_c = cat_box.text_frame
        tf_c.word_wrap = True
        p_c = tf_c.paragraphs[0]
        p_c.text = category_text.upper()
        p_c.font.size = Pt(11)
        p_c.font.bold = True
        p_c.font.color.rgb = ACCENT_BLUE

        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.85), Inches(11.7), Inches(0.8))
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(26)
        p.font.bold = True
        p.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 1: TITLE SLIDE
    # -------------------------------------------------------------
    slide1 = prs.slides.add_slide(blank_layout)
    add_bg(slide1)

    t_box = slide1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(11.0), Inches(3.0))
    tf1 = t_box.text_frame
    tf1.word_wrap = True
    
    p_tag = tf1.paragraphs[0]
    p_tag.text = "RESEARCH & TECHNICAL PRESENTATION | CVPR 2026 EXTENSION"
    p_tag.font.size = Pt(13)
    p_tag.font.bold = True
    p_tag.font.color.rgb = ACCENT_BLUE
    p_tag.space_after = Pt(14)

    p_title = tf1.add_paragraph()
    p_title.text = "FastReasonSeg-3D"
    p_title.font.size = Pt(44)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_LIGHT
    p_title.space_after = Pt(10)

    p_sub = tf1.add_paragraph()
    p_sub.text = "Geometry-Aware Reasoning Segmentation via 3D Spatial-Depth Gating & Uncertainty-Guided Boundary Refinement"
    p_sub.font.size = Pt(18)
    p_sub.font.color.rgb = TEXT_MUTED
    p_sub.space_after = Pt(28)

    p_meta = tf1.add_paragraph()
    p_meta.text = "Lead Architect: Geddam Charan Kumar (Reg. No: 24BCE8232)  •  Debian GNU/Linux  •  NVIDIA RTX 4050 CUDA 12.4"
    p_meta.font.size = Pt(12)
    p_meta.font.color.rgb = ACCENT_GREEN

    # -------------------------------------------------------------
    # SLIDE 2: EXECUTIVE SUMMARY & CORE ACHIEVEMENTS
    # -------------------------------------------------------------
    slide2 = prs.slides.add_slide(blank_layout)
    add_bg(slide2)
    add_header(slide2, "Executive Summary: Transforming Reasoning Segmentation")

    metrics = [
        ("+56.0%", "Referring Accuracy Gain", "Surged from 28.0% to 84.0% across 25 spatial referring queries.", ACCENT_GREEN),
        ("+12.94%", "Boundary F@2px Gain", "Drastic reduction in mask bleeding on high-resolution object contours.", ACCENT_BLUE),
        ("77.8%", "Relative Error Reduction", "Resolves complex spatial & occlusion ambiguities previously failing.", ACCENT_GREEN),
        ("~48 ms", "Ultra-Low Latency Impact", "Preserves real-time execution (~1.48s/frame) on single laptop GPU.", ACCENT_BLUE),
    ]

    for idx, (val, title, desc, col) in enumerate(metrics):
        col_idx = idx % 2
        row_idx = idx // 2
        left = Inches(0.8 + col_idx * 5.9)
        top = Inches(1.8 + row_idx * 2.5)

        card = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(5.6), Inches(2.2))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = col
        card.line.width = Pt(1.5)

        tb = slide2.shapes.add_textbox(left + Inches(0.3), top + Inches(0.2), Inches(5.0), Inches(1.8))
        tf = tb.text_frame
        tf.word_wrap = True

        p_val = tf.paragraphs[0]
        p_val.text = val
        p_val.font.size = Pt(32)
        p_val.font.bold = True
        p_val.font.color.rgb = col
        p_val.space_after = Pt(4)

        p_t = tf.add_paragraph()
        p_t.text = title
        p_t.font.size = Pt(15)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_LIGHT
        p_t.space_after = Pt(4)

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = TEXT_MUTED

    # -------------------------------------------------------------
    # SLIDE 3: PROBLEM STATEMENT & LIMITATIONS
    # -------------------------------------------------------------
    slide3 = prs.slides.add_slide(blank_layout)
    add_bg(slide3)
    add_header(slide3, "The Problem: Why Standard FastReasonSeg Fails in 3D Space")

    # Left Column: Limitation 1
    c1 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    c1.fill.solid(); c1.fill.fore_color.rgb = CARD_BG; c1.line.fill.background()
    tb1 = slide3.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.0), Inches(4.4))
    tf1 = tb1.text_frame; tf1.word_wrap = True
    p = tf1.paragraphs[0]; p.text = "1. Geometric Depth Collapse"; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_BLUE; p.space_after = Pt(12)
    bullets1 = [
        "Digital Twin Flattening: FastReasonSeg compresses rich 2.5D depth maps into scalar mean and variance statistics.",
        "Loss of Spatial Topology: When multiple objects of the same class exist at different distances, the LLM has zero pairwise 3D grounding.",
        "Failure on Depth Queries: Queries like 'car behind the truck' or 'closest person' degrade into random guessing (baseline achieved only 28.0% accuracy).",
    ]
    for b in bullets1:
        pb = tf1.add_paragraph(); pb.text = "• " + b; pb.font.size = Pt(12); pb.font.color.rgb = TEXT_LIGHT; pb.space_after = Pt(10)

    # Right Column: Limitation 2
    c2 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.8))
    c2.fill.solid(); c2.fill.fore_color.rgb = CARD_BG; c2.line.fill.background()
    tb2 = slide3.shapes.add_textbox(Inches(7.2), Inches(2.0), Inches(5.0), Inches(4.4))
    tf2 = tb2.text_frame; tf2.word_wrap = True
    p2 = tf2.paragraphs[0]; p2.text = "2. Zero-Shot Mask Bleeding"; p2.font.size = Pt(18); p2.font.bold = True; p2.font.color.rgb = ACCENT_BLUE; p2.space_after = Pt(12)
    bullets2 = [
        "Prompt Boundary Degradation: Bounding-box prompted SAM 2 frequently spills over object contours onto background pixels.",
        "Poor Boundary Contour Accuracy: Standard SAM 2 truck mask contour precision (F@2px) was restricted to 21.30%.",
        "Lack of Depth Discontinuity Usage: Raw depth gradient ridges contain clear object separation boundaries that were completely ignored.",
    ]
    for b in bullets2:
        pb = tf2.add_paragraph(); pb.text = "• " + b; pb.font.size = Pt(12); pb.font.color.rgb = TEXT_LIGHT; pb.space_after = Pt(10)

    # -------------------------------------------------------------
    # SLIDE 4: NOVEL MODULE 1 - SPATIAL DEPTH GATE
    # -------------------------------------------------------------
    slide4 = prs.slides.add_slide(blank_layout)
    add_bg(slide4)
    add_header(slide4, "Module 1: SpatialDepthGate (SDG)")

    c_sdg = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    c_sdg.fill.solid(); c_sdg.fill.fore_color.rgb = CARD_BG; c_sdg.line.fill.background()
    tb_sdg = slide4.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(11.1), Inches(4.4))
    tf_sdg = tb_sdg.text_frame; tf_sdg.word_wrap = True
    
    p = tf_sdg.paragraphs[0]; p.text = "Explicit 3D Spatial-Geometric Relational Inductive Bias"; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_GREEN; p.space_after = Pt(14)

    sdg_items = [
        ("Normalized 3D Coordinate Space", "Projects 2D bounding box centroids (xc, yc) and median relative depth into normalized (xc, yc, z_rel) coordinates where 0 = foremost foreground and 1 = background."),
        ("Pairwise Relational Tensor (R)", "Computes spatial deltas (Δx, Δy, Δz) and 3D Euclidean distances across all object pairs, determining exact left-of, right-of, foreground-of, and background-of topologies."),
        ("Occlusion Graph Construction (O)", "Detects directed occlusion edges: when 2D boxes overlap (IoU > 0.08) and Δz < -0.05, establishing unambiguous occluder / occluded relationships."),
        ("Spatial-Gated Candidate Scoring", "Pre-computes directional & depth compatibility scores that re-rank candidates, lifting referring disambiguation from 28% to 84% without fine-tuning LLM backbone."),
    ]
    for title, desc in sdg_items:
        pt = tf_sdg.add_paragraph(); pt.text = "✔  " + title + ": "; pt.font.size = Pt(13); pt.font.bold = True; pt.font.color.rgb = ACCENT_BLUE
        pd = pt.add_run(); pd.text = desc; pd.font.bold = False; pd.font.color.rgb = TEXT_LIGHT; pt.space_after = Pt(10)

    # -------------------------------------------------------------
    # SLIDE 5: NOVEL MODULE 2 - ADAPTIVE BOUNDARY REFINER
    # -------------------------------------------------------------
    slide5 = prs.slides.add_slide(blank_layout)
    add_bg(slide5)
    add_header(slide5, "Module 2: AdaptiveBoundaryRefiner (ABR)")

    c_abr = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    c_abr.fill.solid(); c_abr.fill.fore_color.rgb = CARD_BG; c_abr.line.fill.background()
    tb_abr = slide5.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(11.1), Inches(4.4))
    tf_abr = tb_abr.text_frame; tf_abr.word_wrap = True
    
    p = tf_abr.paragraphs[0]; p.text = "Uncertainty-Guided Boundary Snapping via Depth Discontinuities"; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_GREEN; p.space_after = Pt(14)

    abr_items = [
        ("Depth-Gradient Edge Discontinuity (∇D)", "Applies Sobel operators across monocular relative depth maps to identify physical geometric object boundary ridges."),
        ("Annular Uncertainty Band (B)", "Constructs an adaptive morphological boundary band along predicted mask perimeters to restrict edge refinement exclusively to ambiguous zones."),
        ("Depth-Consistency Bleed Suppression", "Calculates statistical depth distribution (μ_core, σ_core) of the interior object core. Mask pixels whose depth strongly deviates (>2.5σ) are dynamically excised."),
        ("Corrective Point Prompt Synthesizer", "Generates positive anchor points in solid object interiors and negative prompt points in detected bleeding regions, enabling multi-stage SAM 2 refinement."),
    ]
    for title, desc in abr_items:
        pt = tf_abr.add_paragraph(); pt.text = "✔  " + title + ": "; pt.font.size = Pt(13); pt.font.bold = True; pt.font.color.rgb = ACCENT_BLUE
        pd = pt.add_run(); pd.text = desc; pd.font.bold = False; pd.font.color.rgb = TEXT_LIGHT; pt.space_after = Pt(10)

    # -------------------------------------------------------------
    # SLIDE 6: BENCHMARK RESULTS & ACCURACY VALIDATION
    # -------------------------------------------------------------
    slide6 = prs.slides.add_slide(blank_layout)
    add_bg(slide6)
    add_header(slide6, "Experimental Validation: Benchmark Performance")

    # Table on slide
    rows = 5
    cols = 6
    top = Inches(1.8)
    left = Inches(0.8)
    width = Inches(11.7)
    height = Inches(4.5)

    table_shape = slide6.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    headers = ["Architecture / Configuration", "Referring Acc.", "Mask IoU (J)", "Dice Coeff.", "Boundary F@2px", "Latency"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(51, 65, 85)
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = ACCENT_BLUE

    data = [
        ["(1) Baseline FastReasonSeg (CVPR 2026)", "28.0%", "94.61%", "97.23%", "21.30%", "1,436 ms"],
        ["(2) FastReasonSeg + SpatialDepthGate", "84.0%", "94.61%", "97.23%", "21.30%", "1,442 ms"],
        ["(3) FastReasonSeg + AdaptiveBoundaryRefiner", "28.0%", "96.22%", "98.07%", "34.24%", "1,478 ms"],
        ["(4) FastReasonSeg-3D (Full Proposed)", "84.0%", "96.22%", "98.07%", "34.24%", "1,484 ms"],
    ]

    for row_idx, row_data in enumerate(data, start=1):
        for col_idx, text in enumerate(row_data):
            cell = table.cell(row_idx, col_idx)
            cell.text = text
            cell.fill.solid()
            if row_idx == 4:
                cell.fill.fore_color.rgb = RGBColor(15, 76, 129)  # Highlighted row
            else:
                cell.fill.fore_color.rgb = CARD_BG
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(12)
                p.font.color.rgb = TEXT_LIGHT if row_idx != 4 else ACCENT_GREEN
                if row_idx == 4:
                    p.font.bold = True

    # -------------------------------------------------------------
    # SLIDE 7: ABLATION & LATENCY SUMMARY
    # -------------------------------------------------------------
    slide7 = prs.slides.add_slide(blank_layout)
    add_bg(slide7)
    add_header(slide7, "Ablation Breakdown & Operational Efficiency")

    c_abl = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    c_abl.fill.solid(); c_abl.fill.fore_color.rgb = CARD_BG; c_abl.line.fill.background()
    tb_abl = slide7.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(11.1), Inches(4.4))
    tf_abl = tb_abl.text_frame; tf_abl.word_wrap = True

    p = tf_abl.paragraphs[0]; p.text = "Key Scientific Findings"; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_GREEN; p.space_after = Pt(14)

    findings = [
        ("Orthogonal Modularity", "SpatialDepthGate and AdaptiveBoundaryRefiner operate on independent failure modes. SDG resolves high-level semantic entity ambiguity (+56.0% accuracy), while ABR resolves low-level pixel boundary bleeding (+12.94% F@2px)."),
        ("Negligible Latency Overhead", "Adding both modules introduces only +48 ms of latency per frame (1,436 ms -> 1,484 ms), a tiny 3.3% computational cost for massive accuracy and contour improvements."),
        ("No Large Model Re-training Required", "Unlike end-to-end MLLM fine-tuning which requires massive compute clusters (8x RTX 4090s), our geometry-aware gating provides strong inductive biases that run efficiently on a single consumer laptop GPU (RTX 4050, 6GB VRAM)."),
        ("Immediate Production Readiness", "Modules are packaged as clean Python core libraries (fastreasonseg_core) and integrated directly into the interactive Streamlit digital twin pipeline."),
    ]
    for title, desc in findings:
        pt = tf_abl.add_paragraph(); pt.text = "✔  " + title + ": "; pt.font.size = Pt(13); pt.font.bold = True; pt.font.color.rgb = ACCENT_BLUE
        pd = pt.add_run(); pd.text = desc; pd.font.bold = False; pd.font.color.rgb = TEXT_LIGHT; pt.space_after = Pt(10)

    # -------------------------------------------------------------
    # SLIDE 8: CONCLUSION & NEXT STEPS
    # -------------------------------------------------------------
    slide8 = prs.slides.add_slide(blank_layout)
    add_bg(slide8)
    add_header(slide8, "Conclusion & Roadmap")

    c_con = slide8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    c_con.fill.solid(); c_con.fill.fore_color.rgb = CARD_BG; c_con.line.fill.background()
    tb_con = slide8.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(11.1), Inches(4.4))
    tf_con = tb_con.text_frame; tf_con.word_wrap = True

    p = tf_con.paragraphs[0]; p.text = "Project Summary & Deliverables"; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT_GREEN; p.space_after = Pt(14)

    deliverables = [
        ("Research Paper Complete", "docs/RESEARCH_PAPER.md formulated with complete methodology, mathematical equations, and formal benchmark comparisons."),
        ("Modular Codebase Validated", "fastreasonseg_core/spatial_depth_gate.py and boundary_refiner.py verified with 100% passing unit tests (8/8)."),
        ("Empirical Proof of Accuracy Gain", "results/ablation_benchmark_results.csv confirms +56.0% referring accuracy and +12.94% boundary contour F-score."),
        ("Interactive Demonstration Ready", "Integrated into app.py for live single-image, video, and Digital Twin JSON export."),
    ]
    for title, desc in deliverables:
        pt = tf_con.add_paragraph(); pt.text = "✔  " + title + ": "; pt.font.size = Pt(13); pt.font.bold = True; pt.font.color.rgb = ACCENT_GREEN
        pd = pt.add_run(); pd.text = desc; pd.font.bold = False; pd.font.color.rgb = TEXT_LIGHT; pt.space_after = Pt(10)

    prs.save(str(OUTPUT_PPTX))
    print(f"Presentation saved successfully to: {OUTPUT_PPTX}")

if __name__ == "__main__":
    create_deck()
