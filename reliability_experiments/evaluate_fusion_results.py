import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_coco_results(gt_path, pred_path, method_name=None):
    """
    Evaluate a fusion output (COCO-style JSON) using COCOeval.
    Returns a dict with mAP@[.5:.95], Precision@0.5, Recall@0.5, F1, runtime.
    """
    start_time = time.time()

    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract COCO stats
    stats = coco_eval.stats
    mAP = float(stats[0])   # IoU=0.50:0.95
    AP50 = float(stats[1])  # IoU=0.50
    AR100 = float(stats[8]) # AR@100

    # Approximate Precision@0.5, Recall@0.5, F1
    # These can be derived from the detailed precision-recall curve in coco_eval.eval['precision']
    precisions = coco_eval.eval['precision'][0, :, 0, 0, -1]  # iou=0.5, class-agnostic
    recalls = coco_eval.params.recThrs
    mean_prec = np.mean(precisions[precisions > -1])
    mean_recall = np.mean(recalls)

    F1 = 2 * (mean_prec * mean_recall) / (mean_prec + mean_recall + 1e-6)

    runtime = time.time() - start_time

    return {
        "Method": method_name or os.path.basename(pred_path),
        "mAP@[.5:.95]": round(mAP, 3),
        "Prec@0.5": round(mean_prec, 3),
        "Rec@0.5": round(mean_recall, 3),
        "F1": round(F1, 3),
        "Time (s)": round(runtime, 2)
    }

def evaluate_all(gt_path, results_dir, save_csv="fusion_summary.csv"):
    """
    Evaluate all JSON result files in a directory and export a summary table.
    """
    results = []
    for file in os.listdir(results_dir):
        if not file.endswith(".json"):
            continue
        pred_path = os.path.join(results_dir, file)
        print(f"🔍 Evaluating {file} ...")
        metrics = evaluate_coco_results(gt_path, pred_path, method_name=file.replace("out_", "").replace(".json", ""))
        results.append(metrics)

    df = pd.DataFrame(results)
    print("\n📊 Summary Table:\n", df)
    df.to_csv(save_csv, index=False)
    print(f"\n✅ Results saved to {save_csv}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fusion outputs (COCO JSON format).")
    parser.add_argument("--gt", required=True, help="Path to COCO ground truth JSON (e.g., instances_val2017.json)")
    parser.add_argument("--results_dir", required=True, help="Directory containing fusion output JSON files")
    parser.add_argument("--save_csv", default="fusion_summary.csv", help="Path to save summary CSV")
    args = parser.parse_args()

    evaluate_all(args.gt, args.results_dir, args.save_csv)
