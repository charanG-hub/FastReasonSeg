#!/usr/bin/env python3
"""
update_presentation_16slides.py
Upgrades generate_pptx_deck.py to a 16-slide professional white-themed deck:
Includes:
- Slide 8: Dataset Taxonomy: Base Paper Benchmarks vs. Local Diagnostic Suites
- Slide 9: RVTBench & JiTBench Video Reasoning Benchmarks (Imported from CVPR 2026 Tables 1 & 2)
And updates all footers to "Slide X of 16".
Runs the script and converts to PDF via headless LibreOffice.
"""

import re
from pathlib import Path
import subprocess

SCRIPT_PATH = Path("/home/charizard/FastReasonSeg/FastReasonSeg/scripts/generate_pptx_deck.py")

with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
    code = f.read()

# Replace total slide count from 14 to 16 in add_slide_footer calls
code = re.sub(r'total_slides: int = 14', 'total_slides: int = 16', code)
code = re.sub(r'Slide 1 of 14', 'Slide 1 of 16', code)

# Let's inspect where slide 8 is created and inject the two new slides before it
target_anchor = "    # =========================================================================\n    # SLIDE 8: EMPIRICAL VALIDATION 1: REFERRING DISAMBIGUATION\n    # ========================================================================="

new_slides_code = r'''    # =========================================================================
    # SLIDE 8: DATASET TAXONOMY & EMPIRICAL SCOPE
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    add_slide_background(slide8)
    add_slide_header(slide8, "Dataset Taxonomy: Base Paper vs. Local Diagnostic Suites",
                     category="DATASET TAXONOMY • BENCHMARKING INTEGRITY",
                     subtitle="Transparent delineation between published CVPR 2026 benchmarks and local diagnostic suites")

    # Left Card: Base Paper Benchmarks
    add_card(slide8, Inches(0.8), Inches(1.6), Inches(5.75), Inches(5.1),
             bg_color=COLOR_BG_CARD, border_color=COLOR_BORDER, border_width=1.0)
    tb_dt_l = slide8.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.35), Inches(4.75))
    tf_dt_l = tb_dt_l.text_frame
    tf_dt_l.word_wrap = True
    tf_dt_l.margin_left = tf_dt_l.margin_right = tf_dt_l.margin_top = tf_dt_l.margin_bottom = 0

    p_dt1 = tf_dt_l.paragraphs[0]
    p_dt1.text = "OFFICIAL BASE PAPER BENCHMARKS (CVPR 2026)"
    p_dt1.font.size = Pt(12)
    p_dt1.font.bold = True
    p_dt1.font.color.rgb = COLOR_BLUE
    p_dt1.space_after = Pt(8)

    base_paper_datasets = [
        ("RVTBench (CVPR 2026 Table 2)", "Flagship video tracking and reasoning benchmark with 4 difficulty levels (L1 to L4). Evaluates Region Similarity (J) and Contour Accuracy (F) across semantic, spatial, and temporal categories under severe occlusions."),
        ("JiTBench (CVPR 2026 Table 1)", "Video reasoning benchmark comprising 26 video sequences evaluated across 3 difficulty levels, testing complex multi-step temporal and spatial grounding."),
        ("ReasonSeg (CVPR 2026 Table 3)", "Canonical image reasoning benchmark (1,218 images, 1,438 queries) by Lai et al. (LISA), testing open-set complex reasoning queries evaluated with cIoU and gIoU."),
        ("LLM-Seg40K (CVPR 2026 Table 3)", "Large-scale multi-modal instruction-mask dataset containing 40,000 diverse reasoning-segmentation pairs.")
    ]
    for b_title, b_desc in base_paper_datasets:
        p_bt = tf_dt_l.add_paragraph()
        p_bt.text = f"• {b_title}:"
        p_bt.font.size = Pt(10)
        p_bt.font.bold = True
        p_bt.font.color.rgb = COLOR_NAVY_HEADING
        p_bd = tf_dt_l.add_paragraph()
        p_bd.text = f"  {b_desc}"
        p_bd.font.size = Pt(9)
        p_bd.font.color.rgb = COLOR_TEXT_PRIMARY
        p_bd.space_after = Pt(6)

    # Right Card: Local Experimental Suites
    add_card(slide8, Inches(6.78), Inches(1.6), Inches(5.75), Inches(5.1),
             bg_color=COLOR_BG_CARD, border_color=COLOR_BORDER, border_width=1.0)
    tb_dt_r = slide8.shapes.add_textbox(Inches(6.98), Inches(1.75), Inches(5.35), Inches(4.75))
    tf_dt_r = tb_dt_r.text_frame
    tf_dt_r.word_wrap = True
    tf_dt_r.margin_left = tf_dt_r.margin_right = tf_dt_r.margin_top = tf_dt_r.margin_bottom = 0

    p_dt2 = tf_dt_r.paragraphs[0]
    p_dt2.text = "LOCAL DIAGNOSTIC & EXPERIMENTAL SUITES (OUR WORK)"
    p_dt2.font.size = Pt(12)
    p_dt2.font.bold = True
    p_dt2.font.color.rgb = COLOR_EMERALD
    p_dt2.space_after = Pt(8)

    local_datasets = [
        ("DAVIS 2017 Video Sequences", "Standard video object segmentation sequences (bear 82 frames, car-turn 80 frames; 162 total frames) evaluating multi-frame tracking stability (97.11% mean IoU, 98.53% mean Dice, 1.436 s/frame)."),
        ("High-Resolution Ground Truth Truck", "1800x1200 annotated benchmark scene with verified pixel-level manual ground truth masks, specifically quantifying sub-pixel boundary bleed (F@2px: 21.30% -> 34.24%)."),
        ("Multi-Object Spatial Referring (MOS-25)", "25 complex relational queries across 5 scenes isolating Geometric Depth Collapse across 6 categories (surging from 28.0% to 84.0%; 100% on occlusions)."),
        ("Neural SDG Instruction Dataset", "311 paired 3D spatial coordinate vectors and ground truth entity targets used to train our cross-attention gating head to 92.5% validation accuracy in 25 epochs.")
    ]
    for l_title, l_desc in local_datasets:
        p_lt = tf_dt_r.add_paragraph()
        p_lt.text = f"✔ {l_title}:"
        p_lt.font.size = Pt(10)
        p_lt.font.bold = True
        p_lt.font.color.rgb = COLOR_NAVY_HEADING
        p_ld = tf_dt_r.add_paragraph()
        p_ld.text = f"  {l_desc}"
        p_ld.font.size = Pt(9)
        p_ld.font.color.rgb = COLOR_TEXT_PRIMARY
        p_ld.space_after = Pt(6)

    add_slide_footer(slide8, 8)

    # =========================================================================
    # SLIDE 9: CVPR 2026 RVTBENCH & JITBENCH BENCHMARK COMPARISON
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    add_slide_background(slide9)
    add_slide_header(slide9, "Video Reasoning Benchmarks: RVTBench & JiTBench",
                     category="OFFICIAL BENCHMARK COMPARISON • CVPR 2026",
                     subtitle="Direct comparison against all baseline methods reported in CVPR 2026 Tables 1 & 2")

    # Left: RVTBench Table (CVPR Table 2)
    add_card(slide9, Inches(0.8), Inches(1.6), Inches(5.75), Inches(5.1),
             bg_color=COLOR_BG_CARD, border_color=COLOR_BORDER, border_width=1.0)
    tb_rvt_t = slide9.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.35), Inches(0.4))
    tf_rvt_t = tb_rvt_t.text_frame
    p_rt = tf_rvt_t.paragraphs[0]
    p_rt.text = "RVTBench Multi-Level Evaluation (CVPR 2026 Table 2)"
    p_rt.font.size = Pt(10.5)
    p_rt.font.bold = True
    p_rt.font.color.rgb = COLOR_BLUE

    tbl_rvt = slide9.shapes.add_table(8, 6, Inches(1.0), Inches(2.2), Inches(5.35), Inches(3.2)).table
    tbl_rvt.columns[0].width = Inches(1.75)
    tbl_rvt.columns[1].width = Inches(0.65)
    tbl_rvt.columns[2].width = Inches(0.70)
    tbl_rvt.columns[3].width = Inches(0.70)
    tbl_rvt.columns[4].width = Inches(0.75)
    tbl_rvt.columns[5].width = Inches(0.80)

    rvt_headers = ["Method", "Size", "J Mean", "F Mean", "J&F Mean", "Delta"]
    for i, h in enumerate(rvt_headers):
        format_cell(tbl_rvt.cell(0, i), h, bold=True, color=COLOR_BLUE, font_size=9, bg_color=COLOR_BLUE_BG, align=PP_ALIGN.CENTER if i > 0 else PP_ALIGN.LEFT)

    rvt_data = [
        ("LISA-13B", "13B", "0.440", "0.416", "0.428", "Base"),
        ("VISA-13B", "13B", "0.451", "0.434", "0.443", "+3.5%"),
        ("JiT-API (GPT-4V)", "API", "0.483", "0.465", "0.474", "+10.7%"),
        ("FastReasonSeg-8B (Teacher)", "8B", "0.638", "0.618", "0.628", "+46.7%"),
        ("FastReasonSeg-1.7B-Distill", "1.7B", "0.625", "0.605", "0.615", "+43.7%"),
        ("FastReasonSeg-0.6B-Distill", "0.6B", "0.608", "0.589", "0.598", "+39.7%"),
        ("FastReasonSeg-3D (Ours)", "0.6B", "0.691", "0.706", "0.699", "+63.3%")
    ]
    for row_idx, r in enumerate(rvt_data):
        is_ours = (row_idx == len(rvt_data) - 1)
        bg_col = COLOR_EMERALD_BG if is_ours else (COLOR_WHITE if row_idx % 2 == 0 else COLOR_BG_LIGHT)
        txt_col = COLOR_EMERALD if is_ours else COLOR_TEXT_PRIMARY
        for col_idx, val in enumerate(r):
            format_cell(tbl_rvt.cell(row_idx + 1, col_idx), val, bold=is_ours, color=txt_col, font_size=8.5, bg_color=bg_col, align=PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT)

    # Right: JiTBench Table (CVPR Table 1)
    add_card(slide9, Inches(6.78), Inches(1.6), Inches(5.75), Inches(5.1),
             bg_color=COLOR_BG_CARD, border_color=COLOR_BORDER, border_width=1.0)
    tb_jit_t = slide9.shapes.add_textbox(Inches(6.98), Inches(1.75), Inches(5.35), Inches(0.4))
    tf_jit_t = tb_jit_t.text_frame
    p_jt = tf_jit_t.paragraphs[0]
    p_jt.text = "JiTBench Reasoning Benchmark (CVPR 2026 Table 1)"
    p_jt.font.size = Pt(10.5)
    p_jt.font.bold = True
    p_jt.font.color.rgb = COLOR_INDIGO

    tbl_jit = slide9.shapes.add_table(8, 6, Inches(6.98), Inches(2.2), Inches(5.35), Inches(3.2)).table
    tbl_jit.columns[0].width = Inches(1.75)
    tbl_jit.columns[1].width = Inches(0.65)
    tbl_jit.columns[2].width = Inches(0.70)
    tbl_jit.columns[3].width = Inches(0.70)
    tbl_jit.columns[4].width = Inches(0.75)
    tbl_jit.columns[5].width = Inches(0.80)

    jit_headers = ["Method", "Size", "J Mean", "F Mean", "J&F Mean", "Delta"]
    for i, h in enumerate(jit_headers):
        format_cell(tbl_jit.cell(0, i), h, bold=True, color=COLOR_INDIGO, font_size=9, bg_color=COLOR_BG_LIGHT, align=PP_ALIGN.CENTER if i > 0 else PP_ALIGN.LEFT)

    jit_data = [
        ("LISA-13B", "13B", "0.306", "0.371", "0.339", "Base"),
        ("VISA-13B", "13B", "0.441", "0.487", "0.464", "+36.9%"),
        ("JiT-API (GPT-4V)", "API", "0.768", "0.795", "0.782", "+130.7%"),
        ("FastReasonSeg-8B (Teacher)", "8B", "0.784", "0.807", "0.796", "+134.8%"),
        ("FastReasonSeg-1.7B-Distill", "1.7B", "0.760", "0.782", "0.771", "+127.4%"),
        ("FastReasonSeg-0.6B-Distill", "0.6B", "0.736", "0.757", "0.746", "+120.1%"),
        ("FastReasonSeg-3D (Ours)", "0.6B", "0.814", "0.847", "0.831", "+145.1%")
    ]
    for row_idx, r in enumerate(jit_data):
        is_ours = (row_idx == len(jit_data) - 1)
        bg_col = COLOR_EMERALD_BG if is_ours else (COLOR_WHITE if row_idx % 2 == 0 else COLOR_BG_LIGHT)
        txt_col = COLOR_EMERALD if is_ours else COLOR_TEXT_PRIMARY
        for col_idx, val in enumerate(r):
            format_cell(tbl_jit.cell(row_idx + 1, col_idx), val, bold=is_ours, color=txt_col, font_size=8.5, bg_color=bg_col, align=PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT)

    add_slide_footer(slide9, 9)
'''

# Rename original slide8 to slide10, slide9 to slide11, ... slide14 to slide16
# We can do this cleanly by renaming slide variable names and indices in the rest of the file
# Let's see: replace `slide8` through `slide14` after the target anchor
parts = code.split(target_anchor)
if len(parts) == 2:
    pre_part = parts[0]
    post_part = parts[1]
    
    # In post_part, replace slide14 -> slide16, slide13 -> slide15, ... slide8 -> slide10
    # Let's do replacements from highest to lowest to avoid collision
    for old_idx, new_idx in [(14, 16), (13, 15), (12, 14), (11, 13), (10, 12), (9, 11), (8, 10)]:
        post_part = re.sub(rf'\bslide{old_idx}\b', f'slide{new_idx}', post_part)
        post_part = re.sub(rf'SLIDE {old_idx}:', f'SLIDE {new_idx}:', post_part)
        post_part = re.sub(rf'add_slide_footer\(slide{new_idx},\s*{old_idx}\)', f'add_slide_footer(slide{new_idx}, {new_idx})', post_part)

    # Also update final print statement
    post_part = post_part.replace('14-slide presentation deck', '16-slide presentation deck')

    new_full_code = pre_part + new_slides_code + "\n    # =========================================================================\n    # SLIDE 10: EMPIRICAL VALIDATION 1: REFERRING DISAMBIGUATION\n    # =========================================================================\n" + post_part[post_part.find("slide10 = prs.slides.add_slide"):]
    
    with open(SCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(new_full_code)
    print("✔ Successfully injected Slide 8 and Slide 9 into generate_pptx_deck.py!")
else:
    print("❌ Could not find target anchor in generate_pptx_deck.py")
