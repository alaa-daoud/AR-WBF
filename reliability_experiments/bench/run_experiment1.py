import sys
import os

# Ensure fusion_bench is in the module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ENSEMBLE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ENSEMBLE_PATH)
import argparse
import yaml
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

from utils.io import save_to_coco_json
#from utils.loader import load_detector_preds_csv

from fusion import nms, soft_nms, cluster_nms
from fusion.aar_wbf import weighted_boxes_fusion_adaptive1
#from ensemble_boxes_wbf import weighted_boxes_fusion
from ensemble_boxes_wbf_experimental import (
    weighted_boxes_fusion_experimental as wbf_baseline
)
from ensemble_boxes_wbf import weighted_boxes_fusion as wbf_per_box

METHODS = {
    "WBF": "baseline",
    "AR-WBF": "adaptive",
    "NMS": nms,
    "SoftNMS": soft_nms,
    "ClusterNMS": cluster_nms,
}
def load_detector_preds_csv(csv_paths, column_map):
    """
    Reads detector predictions from multiple CSV files and returns
    fusion-compatible format: per-image grouped boxes, scores, labels.
    Each image_id maps to a list of per-detector predictions.
    """
    from collections import defaultdict
    import pandas as pd

    # Each image_id maps to a list of N detector predictions
    image_data = defaultdict(lambda: ([], [], []))  # image_id → (boxes_list, scores_list, labels_list)

    for detector_idx, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)
        print(f">> DEBUG columns of {csv_path} : {list(df.columns)}")

        # Per-image: image_id → list of predictions for this detector
        per_image = defaultdict(lambda: ([], [], []))

        for _, row in df.iterrows():
            image_id = row[column_map["image_id"]]
            box = [
                float(row[column_map["x1"]]),
                float(row[column_map["y1"]]),
                float(row[column_map["x2"]]),
                float(row[column_map["y2"]]),
            ]
            score = float(row[column_map["score"]])
            label = int(row[column_map["label"]])

            per_image[image_id][0].append(box)
            per_image[image_id][1].append(score)
            per_image[image_id][2].append(label)

        # Now append this detector's predictions to the master dictionary
        for image_id, (b, s, l) in per_image.items():
            image_data[image_id][0].append(b)
            image_data[image_id][1].append(s)
            image_data[image_id][2].append(l)

    return image_data
def run(config):
    coco_gt = COCO(config["coco"]["val_ann"])

    image_predictions = load_detector_preds_csv(
        config["detector_preds"],
        column_map={
            "image_id": "img_id",
            "label": "label",
            "score": "score",
            "x1": "x1",
            "y1": "y1",
            "x2": "x2",
            "y2": "y2"
        }
    )

    image_ids = list(image_predictions.keys())
    results_all = []

    for method in config["methods"]:
        name = method["name"]
        args = method.get("args", {})
        func = METHODS[name]
        print(f"\n⚙️ Running fusion method: {name}")

        all_results = []
        for image_id in tqdm(image_ids):
            boxes_list_raw, scores_list, labels_list = image_predictions[image_id]
            # Load image size from COCO
            #img_info = coco_gt.loadImgs(int(image_id))[0]
            #img_w, img_h = img_info["width"], img_info["height"]
            boxes_list = []
            for model_boxes in boxes_list_raw:
                normalized_boxes = [
                    [b[0] , b[1] , b[2] , b[3] ]
                    for b in model_boxes
                ]
                boxes_list.append(normalized_boxes)
            if not any(len(boxes) > 0 for boxes in boxes_list):
                continue

            if name == "WBF":
                fused_boxes, fused_scores, fused_labels = wbf_baseline(
                    boxes_list,
                    scores_list,
                    labels_list,
                    weights=args.get("weights"),
                    iou_thr=args.get("iou_thr", 0.55),
                    skip_box_thr=args.get("skip_box_thr", 0.0),
                    conf_type=args.get("conf_type", "avg"),
                    allows_overflow=args.get("allows_overflow", False),
                    skip_checks=False,
                )

            elif name == "AR-WBF":
                fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion_adaptive1(
                    boxes_list,
                    scores_list,
                    labels_list,
                    weights=args.get("weights"),
                    reliability=args.get("reliability"),
                    adaptive_iou=args.get("adaptive_iou", False),
                    image_shape=args.get("image_shape"),
                    delta=args.get("delta", 0.2),
                    iou_thr=args.get("iou_thr", 0.55),
                    skip_box_thr=args.get("skip_box_thr", 0.0),
                    conf_type=args.get("conf_type", "avg"),
                    allows_overflow=args.get("allows_overflow", False),
                )

            else:
                # NMS / Soft-NMS / Cluster-NMS
                fused_boxes, fused_scores, fused_labels = func(
                    boxes_list=boxes_list,
                    scores_list=scores_list,
                    labels_list=labels_list,
                    weights=args.get("weights"),
                    iou_thr=args.get("iou_thr", 0.55),
                    skip_box_thr=args.get("skip_box_thr", 0.0),
                    conf_type=args.get("conf_type", "avg"),
                    allows_overflow=args.get("allows_overflow", False),
                    skip_checks=False,  # ✅ Required for WBF
                )

            fused_scores = [float(s) for s in fused_scores]
            fused_boxes = [[float(x) for x in box] for box in fused_boxes]
            img_info = coco_gt.loadImgs(int(image_id))[0]
            img_w, img_h = img_info["width"], img_info["height"]
            for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
                x1n, y1n, x2n, y2n = box  # normalized
                x_abs = x1n * img_w
                y_abs = y1n * img_h
                w_abs = (x2n - x1n) * img_w
                h_abs = (y2n - y1n) * img_h

                all_results.append({
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "bbox": [float(x_abs), float(y_abs), float(w_abs), float(h_abs)],
                    "score": float(score)
                })

        out_json = f"results_{name.replace(' ', '_').lower()}.json"
        with open(out_json, "w") as f:
            json.dump(all_results, f)

        coco_dt = coco_gt.loadRes(out_json)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats.tolist()
        results_all.append({
            "method": name,
            **{f"metric_{i}": stat for i, stat in enumerate(stats)}
        })

    if "output_csv" in config:
        df = pd.DataFrame(results_all)
        df.to_csv(config["output_csv"], index=False)
        print(f"💾 Results saved to {config['output_csv']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    print(f"📄 Loading config from: {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("🔍 Running with config:")
    print(yaml.dump(config))
    run(config)

if __name__ == "__main__":
    main()
