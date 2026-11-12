#!/usr/bin/env python3
"""
FastReasonSeg project finalizer.
Run from the repository:
    source .venv/bin/activate
    python finalize_fastreasonseg.py --root .

This script consolidates existing results; it does not rerun model inference or
claim official RVTBench evaluation. It computes mask metrics only when real
ground-truth and prediction masks are found.
"""
from __future__ import annotations
import argparse, csv, json, math, statistics, sys
from datetime import datetime
from pathlib import Path

PAPER = {
    "Region Similarity J": {
        "LISA-7B": (.420,.284,.244), "LISA-13B": (.388,.293,.237),
        "VISA-7B": (.479,.398,.354), "VISA-13B": (.507,.431,.384),
        "LLM-Seg": (.307,.177,.153), "SegZero": (.468,.388,.349),
        "CoReS-7B": (.495,.415,.377), "CoReS-13B": (.509,.429,.391),
        "JiT-7B": (.514,.441,.403), "JiT-API": (.792,.766,.747),
        "FastReasonSeg-8B": (.809,.782,.761),
        "FastReasonSeg-1.7B-Distill": (.784,.758,.738),
        "FastReasonSeg-0.6B-Distill": (.760,.733,.714),
    },
    "Contour Accuracy F": {
        "LISA-7B": (.480,.344,.304), "LISA-13B": (.463,.354,.297),
        "VISA-7B": (.492,.442,.424), "VISA-13B": (.530,.475,.455),
        "LLM-Seg": (.386,.314,.252), "SegZero": (.478,.430,.408),
        "CoReS-7B": (.499,.459,.437), "CoReS-13B": (.514,.459,.437),
        "JiT-7B": (.525,.472,.449), "JiT-API": (.806,.801,.777),
        "FastReasonSeg-8B": (.819,.814,.788),
        "FastReasonSeg-1.7B-Distill": (.793,.788,.764),
        "FastReasonSeg-0.6B-Distill": (.768,.763,.739),
    }
}

def read_csv(path):
    if not path.exists(): return []
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        print(f"WARNING: could not read {path}: {e}")
        return []

def nums(rows, key):
    out=[]
    for r in rows:
        try:
            v=float(r.get(key, ""))
            if math.isfinite(v): out.append(v)
        except (TypeError, ValueError): pass
    return out

def mean_or_none(v): return statistics.mean(v) if v else None
def med_or_none(v): return statistics.median(v) if v else None
def fmt(v, n=6): return "N/A" if v is None else f"{v:.{n}f}"

def sequence_summary(name, path):
    rows=read_csv(path)
    if not rows: return None, []
    ious=nums(rows,"iou"); dices=nums(rows,"dice"); times=nums(rows,"elapsed_seconds")
    detected=sum(str(r.get("detected","")).strip().lower()=="true" for r in rows)
    errors=sum(bool(str(r.get("error","")).strip()) for r in rows)
    return {
        "sequence":name,"csv":str(path),"frames":len(rows),
        "iou_count":len(ious),"mean_iou":mean_or_none(ious),"median_iou":med_or_none(ious),
        "dice_count":len(dices),"mean_dice":mean_or_none(dices),"median_dice":med_or_none(dices),
        "timing_count":len(times),"mean_seconds_per_frame":mean_or_none(times),
        "detections":detected,"errors":errors
    }, rows

def mask_metrics(gt_path, pred_path):
    try:
        import numpy as np
        from PIL import Image, ImageFilter
    except Exception as e:
        return {"error":f"NumPy/Pillow unavailable: {e}"}
    try:
        gt_img=Image.open(gt_path).convert("L")
        pr_img=Image.open(pred_path).convert("L")
        if gt_img.size != pr_img.size:
            return {"error":f"size mismatch GT={gt_img.size}, pred={pr_img.size}"}
        gt=np.asarray(gt_img)>0
        pr=np.asarray(pr_img)>0
        inter=int(np.logical_and(gt,pr).sum())
        union=int(np.logical_or(gt,pr).sum())
        gp=int(gt.sum()); pp=int(pr.sum())
        iou=inter/union if union else 1.0
        dice=(2*inter/(gp+pp)) if (gp+pp) else 1.0
        # Boundary F at 2-pixel tolerance. Boundary = mask - eroded(mask);
        # dilation is performed with Pillow MaxFilter for portable dependencies.
        def boundary(a):
            im=Image.fromarray((a.astype("uint8")*255))
            eroded=im.filter(ImageFilter.MinFilter(3))
            return np.logical_and(a, np.asarray(eroded)==0)
        def dilate(a, radius=2):
            im=Image.fromarray((a.astype("uint8")*255))
            k=2*radius+1
            return np.asarray(im.filter(ImageFilter.MaxFilter(k)))>0
        gb=boundary(gt); pb=boundary(pr)
        precision=float(np.logical_and(pb,dilate(gb)).sum())/float(pb.sum()) if pb.sum() else (1.0 if not gb.sum() else 0.0)
        recall=float(np.logical_and(gb,dilate(pb)).sum())/float(gb.sum()) if gb.sum() else (1.0 if not pb.sum() else 0.0)
        f=2*precision*recall/(precision+recall) if precision+recall else 0.0
        return {"gt":str(gt_path),"prediction":str(pred_path),"width":gt_img.width,"height":gt_img.height,
                "gt_pixels":gp,"pred_pixels":pp,"intersection":inter,"union":union,
                "iou_J":iou,"dice":dice,"boundary_precision":precision,"boundary_recall":recall,
                "boundary_F_tolerance_2px":f}
    except Exception as e:
        return {"error":f"{type(e).__name__}: {e}"}

def write_csv(path, rows):
    if not rows: return
    keys=list(dict.fromkeys(k for r in rows for k in r.keys()))
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".",help="repository root")
    args=ap.parse_args()
    root=Path(args.root).expanduser().resolve()
    out=root/"docs"/"recovered_results"
    out.mkdir(parents=True,exist_ok=True)
    print("FASTREASONSEG FINALIZATION")
    print("Root:",root)
    benchmarks=[
        ("Bear",root/"data/bear_benchmark/results.csv"),
        ("Car-turn",root/"data/car_turn_benchmark/results.csv"),
        ("Bear-evaluation",root/"data/bear_evaluation/results.csv"),
    ]
    summaries=[]; all_rows=[]
    for name,path in benchmarks:
        s, rows=sequence_summary(name,path)
        if s:
            summaries.append(s)
            for r in rows:
                rr=dict(r); rr["sequence"]=name; all_rows.append(rr)
            print(f"{name}: frames={s['frames']} IoU={fmt(s['mean_iou'])} Dice={fmt(s['mean_dice'])} time={fmt(s['mean_seconds_per_frame'],3)}s errors={s['errors']}")
    if all_rows:
        write_csv(out/"final_per_frame_results.csv",all_rows)
    write_csv(out/"final_sequence_summary.csv",summaries)

    # Existing single-image truck GT and candidate predictions.
    truck_gt=root/"data/truck_ground_truth_mask.png"
    candidates=[
        root/"data/truck_sam2_refined_mask.png",
        root/"data/truck_refined_mask.png",
        root/"data/truck_clean_mask.png",
        root/"data/truck_mask.png",
    ]
    truck_results=[]
    if truck_gt.exists():
        for pred in candidates:
            if pred.exists():
                r=mask_metrics(truck_gt,pred); r["sample"]="truck_single_image"; truck_results.append(r)
    write_csv(out/"truck_mask_metrics.csv",truck_results)

    # Copy paper comparison values supplied by the project owner into an explicit,
    # clearly labelled reference table. These are paper values, not local results.
    paper_rows=[]
    for metric, methods in PAPER.items():
        for method, vals in methods.items():
            paper_rows.append({"metric":metric,"method":method,"level_1":vals[0],"level_2":vals[1],"level_3":vals[2],
                               "source_status":"paper values supplied by user; not recomputed"})
    write_csv(out/"paper_rvtbench_reference_values.csv",paper_rows)

    lines=["# FastReasonSeg Project Finalization Report","",
           f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}","",
           "> Scope: consolidate existing saved CSV metrics and evaluate available single-image GT/prediction masks. This script does not run model inference, train models, or certify an official RVTBench evaluation.","",
           "## 1. Existing per-sequence results","",
           "| Sequence | CSV rows | IoU rows | Mean IoU | Median IoU | Dice rows | Mean Dice | Mean seconds/frame | Detections | Errors |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['sequence']} | {s['frames']} | {s['iou_count']} | {fmt(s['mean_iou'])} | {fmt(s['median_iou'])} | {s['dice_count']} | {fmt(s['mean_dice'])} | {fmt(s['mean_seconds_per_frame'],3)} | {s['detections']} | {s['errors']} |")
    all_iou=[x for s in summaries for x in nums(read_csv(Path(s["csv"])),"iou")]
    all_dice=[x for s in summaries for x in nums(read_csv(Path(s["csv"])),"dice")]
    all_time=[x for s in summaries for x in nums(read_csv(Path(s["csv"])),"elapsed_seconds")]
    lines += ["","## 2. Combined saved-record summary","",
              f"- Sequences with readable results: **{len(summaries)}**",
              f"- Total CSV rows: **{sum(s['frames'] for s in summaries)}**",
              f"- Valid saved IoU values: **{len(all_iou)}**; mean: **{fmt(mean_or_none(all_iou))}**",
              f"- Valid saved Dice values: **{len(all_dice)}**; mean: **{fmt(mean_or_none(all_dice))}**",
              f"- Valid timing values: **{len(all_time)}**; mean: **{fmt(mean_or_none(all_time),3)} seconds/frame**",
              "",
              "These are arithmetic summaries of existing CSV values. They do not independently verify the original annotations, mask alignment, or metric implementation."]
    lines += ["","## 3. Truck single-image mask evaluation",""]
    if truck_results:
        lines += ["| Prediction mask | IoU / J | Dice | Boundary F (2 px) | GT pixels | Predicted pixels |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in truck_results:
            if "error" in r:
                lines.append(f"| {Path(r.get('prediction','unknown')).name} | ERROR | - | - | - | - |")
            else:
                lines.append(f"| {Path(r['prediction']).name} | {fmt(r['iou_J'])} | {fmt(r['dice'])} | {fmt(r['boundary_F_tolerance_2px'])} | {r['gt_pixels']} | {r['pred_pixels']} |")
        lines.append("")
        lines.append("Boundary F is calculated with a 2-pixel tolerance on the available truck annotation and is not an RVTBench result.")
    else:
        lines += ["No usable truck ground-truth/prediction mask pair found. Expected `data/truck_ground_truth_mask.png` and at least one candidate mask file.",""]
    lines += ["## 4. Paper reference table","",
              "Reference values supplied by the project owner are saved to `paper_rvtbench_reference_values.csv`. They are paper-reported RVTBench values and are not measurements of this local prototype. Local DAVIS results must not be compared as if they were RVTBench results without matching dataset, levels, prompts, and evaluation protocol.","",
              "## 5. Implementation status and honest limitations","",
              "- Existing code/results demonstrate a segmentation prototype with saved frame metrics and motion/mask analyses.",
              "- Existing reports describe frame-wise detection/segmentation; do not claim persistent tracking unless the relevant implementation is verified.",
              "- A small number of video sequences does not establish generalization.",
              "- The full paper-specific teacher training, SFT/GRPO training, and student distillation pipeline remains unimplemented according to the existing project report.",
              "- This script does not fabricate missing metrics and does not label prototype results as official RVTBench.",
              "",
              "## 6. Generated files","",
              "- `final_per_frame_results.csv` — merged per-frame CSV records",
              "- `final_sequence_summary.csv` — sequence-level statistics",
              "- `truck_mask_metrics.csv` — GT-based truck mask IoU, Dice, and 2-pixel boundary F if masks exist",
              "- `paper_rvtbench_reference_values.csv` — paper reference values supplied by the project owner",
              "- `FINAL_PROJECT_STATUS.md` — this report",
              "",
              "## 7. Reproducibility command","",
              "```bash",
              "source .venv/bin/activate",
              "python finalize_fastreasonseg.py --root .",
              "```",""]
    report=out/"FINAL_PROJECT_STATUS.md"
    report.write_text("\n".join(lines),encoding="utf-8")
    print("\nOutputs:")
    for p in [report,out/"final_per_frame_results.csv",out/"final_sequence_summary.csv",out/"truck_mask_metrics.csv",out/"paper_rvtbench_reference_values.csv"]:
        print(f" - {p} {'[created]' if p.exists() else '[not created]'}")
    print("\nFinished consolidation. Official RVTBench status: NOT VERIFIED.")

if __name__=="__main__":
    main()
