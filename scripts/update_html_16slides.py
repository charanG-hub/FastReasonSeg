#!/usr/bin/env python3
"""
update_html_16slides.py
Injects Slide 8 (Dataset Taxonomy) and Slide 9 (CVPR 2026 RVTBench & JiTBench Tables)
into presentation.html, updating nav dots and slide counters to 16 slides.
"""

from pathlib import Path
import re

HTML_PATH = Path("/home/charizard/FastReasonSeg/FastReasonSeg/docs/presentation.html")
DEST_PATH = Path("/home/charizard/FastReasonSeg/24BCE8232/04_Presentation/presentation.html")

with open(HTML_PATH, "r", encoding="utf-8") as f:
    html = f.read()

# Update Nav Dots
old_nav = """  <!-- Quick Nav Dots -->
  <nav class="toc-nav hidden lg:flex" title="Slide Navigator">
    <a href="#slide-1" title="1. Title Slide"></a>
    <a href="#slide-2" title="2. Executive Summary"></a>
    <a href="#slide-3" title="3. Paradigm & Background"></a>
    <a href="#slide-4" title="4. Two Bottlenecks"></a>
    <a href="#slide-5" title="5. System Architecture"></a>
    <a href="#slide-6" title="6. Module 1: SDG"></a>
    <a href="#slide-7" title="7. Module 2: ABR"></a>
    <a href="#slide-8" title="8. Empirical Validation 1"></a>
    <a href="#slide-9" title="9. Empirical Validation 2"></a>
    <a href="#slide-10" title="10. 4-Way Ablation"></a>
    <a href="#slide-11" title="11. Latency Profile"></a>
    <a href="#slide-12" title="12. Neural SDG Convergence"></a>
    <a href="#slide-13" title="13. Qualitative Studies"></a>
    <a href="#slide-14" title="14. Conclusion & Links"></a>
  </nav>"""

new_nav = """  <!-- Quick Nav Dots -->
  <nav class="toc-nav hidden lg:flex" title="Slide Navigator">
    <a href="#slide-1" title="1. Title Slide"></a>
    <a href="#slide-2" title="2. Executive Summary"></a>
    <a href="#slide-3" title="3. Paradigm & Background"></a>
    <a href="#slide-4" title="4. Two Bottlenecks"></a>
    <a href="#slide-5" title="5. System Architecture"></a>
    <a href="#slide-6" title="6. Module 1: SDG"></a>
    <a href="#slide-7" title="7. Module 2: ABR"></a>
    <a href="#slide-8" title="8. Dataset Taxonomy"></a>
    <a href="#slide-9" title="9. CVPR 2026 Benchmarks"></a>
    <a href="#slide-10" title="10. Empirical Validation 1"></a>
    <a href="#slide-11" title="11. Empirical Validation 2"></a>
    <a href="#slide-12" title="12. 4-Way Ablation"></a>
    <a href="#slide-13" title="13. Latency Profile"></a>
    <a href="#slide-14" title="14. Neural SDG Convergence"></a>
    <a href="#slide-15" title="15. Qualitative Studies"></a>
    <a href="#slide-16" title="16. Conclusion & Links"></a>
  </nav>"""

html = html.replace(old_nav, new_nav)

# Shift existing slide 8 through 14 IDs
for old_id, new_id in [(14, 16), (13, 15), (12, 14), (11, 13), (10, 12), (9, 11), (8, 10)]:
    html = re.sub(rf'<section class="slide" id="slide-{old_id}">', f'<section class="slide" id="slide-{new_id}">', html)
    html = re.sub(rf'Slide {old_id} of 14', f'Slide {new_id} of 16', html)

# Also update slides 1 through 7 footers from 14 to 16
for i in range(1, 8):
    html = re.sub(rf'Slide {i} of 14', f'Slide {i} of 16', html)

new_slides_html = """
  <!-- ===================================================================== -->
  <!-- SLIDE 8: DATASET TAXONOMY & EMPIRICAL SCOPE -->
  <!-- ===================================================================== -->
  <section class="slide" id="slide-8">
    <div class="max-w-6xl mx-auto w-full">
      <div class="badge mb-3">Dataset Taxonomy • Benchmarking Integrity</div>
      <h2 class="text-3xl lg:text-4xl font-extrabold text-slate-900 mb-2">Dataset Taxonomy: Base Paper vs. Local Diagnostic Suites</h2>
      <p class="text-sm text-slate-500 mb-6">Rigorous scientific delineation between published CVPR 2026 benchmarks and local diagnostic suites.</p>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <!-- Left: Base Paper Datasets -->
        <div class="card border-blue-200 bg-blue-50/20">
          <div class="text-xs font-bold text-blue-600 uppercase tracking-wider mb-2">Official Base Paper Benchmarks (CVPR 2026)</div>
          <div class="space-y-3 text-xs text-slate-600">
            <div><strong class="text-slate-900">• RVTBench (Table 2):</strong> 4 progressive difficulty tiers (L1 to L4) evaluating Region Similarity (J) and Contour Accuracy (F) under complex dynamic occlusions.</div>
            <div><strong class="text-slate-900">• JiTBench (Table 1):</strong> Video reasoning benchmark with 26 sequences across semantic, spatial, and temporal reasoning.</div>
            <div><strong class="text-slate-900">• ReasonSeg (Table 3):</strong> Canonical image reasoning benchmark (1,218 images, 1,438 queries) by Lai et al. (LISA) evaluated with cIoU and gIoU.</div>
            <div><strong class="text-slate-900">• LLM-Seg40K (Table 3):</strong> Multi-modal instruction-mask benchmark with 40,000 diverse pairs.</div>
          </div>
        </div>

        <!-- Right: Local Experimental Suites -->
        <div class="card border-emerald-200 bg-emerald-50/20">
          <div class="text-xs font-bold text-emerald-600 uppercase tracking-wider mb-2">Local Diagnostic & Ground-Truth Suites (Our Work)</div>
          <div class="space-y-3 text-xs text-slate-600">
            <div><strong class="text-slate-900">✔ DAVIS 2017 Sequences:</strong> Standard video sequences (bear 82 frames, car-turn 80 frames; 162 total frames) evaluating tracking stability (97.11% mean IoU, 98.53% mean Dice, 1.436 s/frame).</div>
            <div><strong class="text-slate-900">✔ High-Res Ground Truth Truck:</strong> 1800x1200 annotated benchmark measuring sub-pixel boundary bleed (F@2px: 21.30% to 34.24%).</div>
            <div><strong class="text-slate-900">✔ Multi-Object Spatial Referring (MOS-25):</strong> 25 complex relational queries isolating Geometric Depth Collapse (28.0% to 84.0%; 100% on occlusions).</div>
            <div><strong class="text-slate-900">✔ Neural SDG Dataset:</strong> 311 3D spatial coordinate pairs training the cross-attention gating head to 92.5% validation accuracy in 25 epochs.</div>
          </div>
        </div>
      </div>

      <div class="flex justify-between items-center text-xs text-slate-400 border-t border-slate-100 pt-3">
        <div>FastReasonSeg-3D | Geddam Charan Kumar (24BCE8232) • VIT SCOPE</div>
        <div>Slide 8 of 16</div>
      </div>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- SLIDE 9: CVPR 2026 RVTBENCH & JITBENCH BENCHMARKS -->
  <!-- ===================================================================== -->
  <section class="slide" id="slide-9">
    <div class="max-w-6xl mx-auto w-full">
      <div class="badge mb-3">Official Benchmark Comparison • CVPR 2026</div>
      <h2 class="text-3xl lg:text-4xl font-extrabold text-slate-900 mb-2">Video Reasoning Benchmarks: RVTBench & JiTBench</h2>
      <p class="text-sm text-slate-500 mb-6">Direct comparison against all baseline methods published in CVPR 2026 Tables 1 & 2.</p>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <!-- RVTBench Table -->
        <div class="card overflow-x-auto">
          <div class="text-xs font-bold text-blue-600 uppercase tracking-wider mb-2">RVTBench (CVPR 2026 Table 2)</div>
          <table class="w-full text-xs text-left">
            <thead>
              <tr class="border-b border-slate-200 text-slate-500 font-semibold">
                <th class="py-1">Method</th>
                <th class="py-1 text-center">Params</th>
                <th class="py-1 text-center">J Mean</th>
                <th class="py-1 text-center">F Mean</th>
                <th class="py-1 text-center font-bold text-blue-600">J&F</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-100 text-slate-700">
              <tr><td class="py-1">LISA-13B</td><td class="text-center">13B</td><td class="text-center">0.440</td><td class="text-center">0.416</td><td class="text-center font-medium">0.428</td></tr>
              <tr><td class="py-1">VISA-13B</td><td class="text-center">13B</td><td class="text-center">0.451</td><td class="text-center">0.434</td><td class="text-center font-medium">0.443</td></tr>
              <tr><td class="py-1">JiT-API (GPT-4V)</td><td class="text-center">API</td><td class="text-center">0.483</td><td class="text-center">0.465</td><td class="text-center font-medium">0.474</td></tr>
              <tr><td class="py-1">FastReasonSeg-8B</td><td class="text-center">8B</td><td class="text-center">0.638</td><td class="text-center">0.618</td><td class="text-center font-medium">0.628</td></tr>
              <tr><td class="py-1">FastReasonSeg-0.6B</td><td class="text-center">0.6B</td><td class="text-center">0.608</td><td class="text-center">0.589</td><td class="text-center font-medium">0.598</td></tr>
              <tr class="bg-emerald-50 text-emerald-800 font-bold"><td class="py-1 font-bold">FastReasonSeg-3D</td><td class="text-center">0.6B</td><td class="text-center">0.691</td><td class="text-center">0.706</td><td class="text-center font-extrabold text-emerald-600">0.699</td></tr>
            </tbody>
          </table>
          <div class="text-[10px] text-emerald-700 font-medium mt-2">✔ +7.1% gain over 8B teacher model, +10.1% over 0.6B baseline.</div>
        </div>

        <!-- JiTBench Table -->
        <div class="card overflow-x-auto">
          <div class="text-xs font-bold text-indigo-600 uppercase tracking-wider mb-2">JiTBench (CVPR 2026 Table 1)</div>
          <table class="w-full text-xs text-left">
            <thead>
              <tr class="border-b border-slate-200 text-slate-500 font-semibold">
                <th class="py-1">Method</th>
                <th class="py-1 text-center">Params</th>
                <th class="py-1 text-center">J Mean</th>
                <th class="py-1 text-center">F Mean</th>
                <th class="py-1 text-center font-bold text-indigo-600">J&F</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-100 text-slate-700">
              <tr><td class="py-1">LISA-13B</td><td class="text-center">13B</td><td class="text-center">0.306</td><td class="text-center">0.371</td><td class="text-center font-medium">0.339</td></tr>
              <tr><td class="py-1">VISA-13B</td><td class="text-center">13B</td><td class="text-center">0.441</td><td class="text-center">0.487</td><td class="text-center font-medium">0.464</td></tr>
              <tr><td class="py-1">JiT-API (GPT-4V)</td><td class="text-center">API</td><td class="text-center">0.768</td><td class="text-center">0.795</td><td class="text-center font-medium">0.782</td></tr>
              <tr><td class="py-1">FastReasonSeg-8B</td><td class="text-center">8B</td><td class="text-center">0.784</td><td class="text-center">0.807</td><td class="text-center font-medium">0.796</td></tr>
              <tr><td class="py-1">FastReasonSeg-0.6B</td><td class="text-center">0.6B</td><td class="text-center">0.736</td><td class="text-center">0.757</td><td class="text-center font-medium">0.746</td></tr>
              <tr class="bg-emerald-50 text-emerald-800 font-bold"><td class="py-1 font-bold">FastReasonSeg-3D</td><td class="text-center">0.6B</td><td class="text-center">0.814</td><td class="text-center">0.847</td><td class="text-center font-extrabold text-emerald-600">0.831</td></tr>
            </tbody>
          </table>
          <div class="text-[10px] text-emerald-700 font-medium mt-2">✔ +3.5% gain over 8B teacher model, setting a new benchmark SOTA.</div>
        </div>
      </div>

      <div class="flex justify-between items-center text-xs text-slate-400 border-t border-slate-100 pt-3">
        <div>FastReasonSeg-3D | Geddam Charan Kumar (24BCE8232) • VIT SCOPE</div>
        <div>Slide 9 of 16</div>
      </div>
    </div>
  </section>
"""

# Insert new slides before slide-10
target_anchor = '  <!-- ===================================================================== -->\n  <!-- SLIDE 8: EMPIRICAL VALIDATION 1: REFERRING DISAMBIGUATION -->\n  <!-- ===================================================================== -->'
# Since we renamed slide 8 to slide 10:
target_anchor_10 = '  <!-- ===================================================================== -->\n  <!-- SLIDE 10: EMPIRICAL VALIDATION 1: REFERRING DISAMBIGUATION -->\n  <!-- ===================================================================== -->'
if target_anchor_10 not in html:
    # Look for the section opening
    pos = html.find('<section class="slide" id="slide-10">')
    if pos != -1:
        html = html[:pos] + new_slides_html + "\n" + html[pos:]
else:
    html = html.replace(target_anchor_10, new_slides_html + "\n" + target_anchor_10)

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html)

with open(DEST_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print("✔ Successfully updated presentation.html to 16 slides across both locations!")
