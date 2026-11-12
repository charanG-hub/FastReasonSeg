# FastReasonSeg-3D: Geometry-Aware Reasoning Segmentation via 3D Spatial-Depth Gating and Uncertainty-Guided Boundary Refinement

> **Course / Program:** Advanced AI & Computer Vision — Capstone Research Project  
> **Student:** Geddam Charan Kumar | **Reg. No.:** 24BCE8232 | **Email:** geddamcharan24@gmail.com  
> **Institution:** School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology (VIT), India  
> **Base Paper:** *Fast Reasoning Segmentation for Images and Videos* — IEEE/CVF CVPR 2026 (Shen & Unberath)  
> **Research Paper:** 17-Page Comprehensive Technical Report (`docs/paper/FastReasonSeg_3D_Research_Paper.pdf`)  
> **Presentation Deck:** 14-Slide Professional White-Themed Deck (`docs/FastReasonSeg_3D_Presentation.pdf` / `.pptx`)  
> **GitHub Repository:** [https://github.com/charanG-hub/FastReasonSeg](https://github.com/charanG-hub/FastReasonSeg)  
> **Kaggle Public Dataset:** [https://www.kaggle.com/datasets/charankumargeddam/fastreasonseg-3d](https://www.kaggle.com/datasets/charankumargeddam/fastreasonseg-3d)  
> **Kaggle Public Notebook:** [https://www.kaggle.com/code/charankumargeddam/fastreasonseg-3d-reasoning-segmentation](https://www.kaggle.com/code/charankumargeddam/fastreasonseg-3d-reasoning-segmentation)  

---

## Executive Summary & Empirical Verification

FastReasonSeg-3D resolves the two fundamental architectural bottlenecks of decoupled reasoning segmentation:
1. **Geometric Depth Collapse:** Dense 2.5D relative depth fields were previously reduced to scalar means ($\mu, \sigma$), losing spatial topology. We introduce **SpatialDepthGate (SDG)**, projecting instances into normalized 3D coordinates $(x_c, y_c, z_{rel})$, building pairwise 3D relational tensors $\mathcal{R}$, and evaluating directed occlusion graphs $\mathcal{O}$.
2. **Zero-Shot Boundary Bleed:** SAM 2 bounding-box prompting suffers from mask leakage across depth discontinuities. We introduce **AdaptiveBoundaryRefiner (ABR)**, detecting boundary entropy across Sobel depth ridges and dynamically synthesizing corrective positive/negative prompt points.

### Empirical Performance Comparison: Baseline vs. FastReasonSeg-3D

| Metric / Dimension | Baseline FastReasonSeg (CVPR 2026) | FastReasonSeg-3D (Ours) | Absolute Gain (Δ) | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Referring Accuracy** | 28.0% | **84.0%** | **+56.0%** | **77.8% Relative Error Reduction** |
| **Occlusion Depth Queries** | 0.0% | **100.0%** | **+100.0%** | **Complete Resolution of Occlusion** |
| **Depth + Spatial Queries** | 20.0% | **100.0%** | **+80.0%** | **5.0× Accuracy Surge** |
| **Boundary Contour $F@2\text{px}$** | 21.30% | **34.24%** | **+12.94%** | **+60.7% Boundary Precision** |
| **Boundary Contour $F@5\text{px}$** | 40.87% | **51.54%** | **+10.67%** | **+26.1% Contour Recall** |
| **Mask IoU (Jaccard Index)** | 94.61% | **96.22%** | **+1.61%** | **Sub-pixel Edge Alignment** |
| **Neural SDG Validation Acc.** | — | **92.50%** | **+92.5%** | **Trained locally in 25 epochs** |
| **Total Pipeline Latency** | 1,436 ms | **1,484 ms** | **+48 ms** | **Ultra-low ~3.3% runtime overhead** |
| **Peak GPU VRAM Footprint** | 4.8 GB | **5.2 GB** | **+0.4 GB** | **100% Consumer Laptop GPU Viable** |

---

## Submission Deliverables & Directory Structure

```
24BCE8232/
├── 01_Base_Paper/
│   ├── Research_Paper/
│   │   ├── FastReasonSeg_3D_Research_Paper.pdf    # 9-page IEEE/CVF publication paper
│   │   ├── main.tex                               # Full LaTeX conference source
│   │   └── figures/                               # 5 publication figures (300 DPI)
│   └── Research_Paper_Overleaf.zip                # Self-contained Overleaf upload package
├── 02_Original_Implementation/
│   ├── Code/
│   │   ├── Notebooks/
│   │   │   └── FastReasonSeg_Project.ipynb        # Original base notebook
│   │   └── Source/
│   │       └── app_baseline.py                    # Original Streamlit perception pipeline
│   └── Results/                                   # Benchmark reference values and evaluation logs
├── 03_Modified_Implementation/
│   ├── Architecture_Diagram/
│   │   └── FastReasonSeg_3D_Architecture.png      # High-resolution architectural schematic
│   ├── Checkpoints/
│   │   └── neural_spatial_depth_gate.pt           # PyTorch trained weights (92.5% Val Acc)
│   ├── Code/
│   │   ├── Notebooks/
│   │   │   └── FastReasonSeg_3D.ipynb             # Runnable Kaggle demonstration notebook
│   │   └── Source/
│   │       ├── fastreasonseg_core/                # Novel SDG, ABR, Digital Twin modules
│   │       └── app.py                             # Interactive application with 3D reasoning
│   ├── Documentation/
│   │   ├── RESEARCH_PAPER.md                      # Publication report with LaTeX math
│   │   └── FINAL_RESEARCH_COMPARISON.md           # Benchmark breakdown
│   ├── Log_Files/
│   │   └── training_metrics.json                  # 25-epoch training progression logs
│   └── Results/
│       ├── ablation_benchmark_results.csv         # 4-way ablation experimental metrics
│       └── truck_adaptive_refined_mask.png        # Sub-pixel refined mask output
├── 04_Presentation/
│   ├── Results_Presentation.pptx                  # Executive widescreen presentation deck
│   ├── Results_Presentation.pdf                   # Exported presentation PDF
│   ├── presentation.html                          # Interactive modern HTML slide deck
│   └── figures/                                   # High-resolution visual figures
├── Student_Research_Report.pdf                     # Standalone 9-page research paper PDF
├── Results_Presentation.pptx                      # Direct submission PPTX presentation
├── Results_Presentation.pdf                       # Direct submission PDF presentation
├── 24BCE8232.zip                                  # Complete academic project zip
└── README.md                                      # Master project documentation
```

---

## Hardware & System Specification

- **Developer:** Geddam Charan Kumar (Reg. No.: 24BCE8232)
- **Workstation:** Arceus Laptop
- **OS:** Debian GNU/Linux 13 (trixie) x86_64 (KDE Plasma 6, Wayland)
- **CPU:** 13th Gen Intel Core i7-13620H (10 cores, 16 threads, up to 4.90 GHz)
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, CUDA 12.4, Driver 550.163.01)
- **Runtimes:** Python 3.11, PyTorch 2.4.0+cu124, OpenCV 4.10, Hugging Face Transformers

---

## Quick Start & Reproduction

```bash
# Clone the repository
git clone https://github.com/charanG-hub/FastReasonSeg.git
cd FastReasonSeg

# Install dependencies
pip install -r requirements.txt

# Run the 4-way ablation validation benchmark
python scripts/validate_novel_modules.py

# Train the Neural SpatialDepthGate on CUDA
python scripts/train_spatial_gate.py

# Launch the interactive FastReasonSeg-3D UI
streamlit run app.py
```
