import os
import json
import pandas as pd
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def load_predictions_csv(pred_path):
    import pandas as pd, os
    df = pd.read_csv(pred_path)
    if "img_id" in df.columns:
        df["image_id"] = df["img_id"].apply(lambda x: int(os.path.splitext(str(x))[0].split('/')[-1].split('.')[0]))
    if {"x1","y1","x2","y2"}.issubset(df.columns):
        df["bbox"] = df[["x1","y1","x2","y2"]].values.tolist()
    results = []
    for _, row in df.iterrows():
        x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
        results.append({
            "image_id": int(row["image_id"]),
            "category_id": int(row["label"]),
            "bbox": [x1, y1, x2-x1, y2-y1],
            "score": float(row["score"])
        })
    tmp_json = pred_path.replace(".csv", "_converted.json")
    import json
    with open(tmp_json, "w") as f:
        json.dump(results, f)
    return tmp_json


def _load_detections(pred_path, coco_gt):
    """
    Load model detections in either CSV or COCO JSON format.
    Returns: COCO results object (list of dicts).
    """
    if pred_path.endswith(".csv"):
        df = pd.read_csv(pred_path)
        results = []
        for _, row in df.iterrows():
            raw_id = str(row["img_id"])
            # Handle cases like "000000123456.jpg" or "val2017/000000123456.jpg"
            img_id_str = os.path.splitext(os.path.basename(raw_id))[0]
            try:
                img_id = int(img_id_str)
            except ValueError:
                continue  # skip bad rows
            results.append({
                "image_id": img_id,
                "category_id": int(row["label"]),
                "bbox": [
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"] - row["x1"]),
                    float(row["y2"] - row["y1"])
                ],
                "score": float(row["score"])
            })
        if not results:
            print(f"⚠️ No valid detections parsed from {pred_path}")
    return coco_gt.loadRes(results)

def evaluate_model(pred_path, gt_path, metric="mAP"):
    """
    Evaluate a model's predictions (CSV or JSON) using COCOeval.
    Supports normalized CSVs like those used in WBF fusion.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import pandas as pd, numpy as np, os

    coco_gt = COCO(gt_path)

    if pred_path.endswith(".csv"):
        df = pd.read_csv(pred_path)
        # Normalize image IDs
        img_ids = []
        for raw in df["img_id"]:
            img_id_str = os.path.splitext(os.path.basename(str(raw)))[0]
            try:
                img_ids.append(int(img_id_str))
            except ValueError:
                img_ids.append(int(float(raw)))  # fallback if numeric

        df["img_id"] = img_ids

        detections = np.zeros((len(df), 7), dtype=np.float64)
        detections[:, 0] = df["img_id"].astype(np.int32)
        x1 = df["x1"].values
        y1 = df["y1"].values
        x2 = df["x2"].values
        y2 = df["y2"].values

        # Convert normalized coords to absolute pixels
        for i, img_id in enumerate(detections[:, 0].astype(np.int32)):
            info = coco_gt.loadImgs(int(img_id))[0]
            w, h = info["width"], info["height"]
            detections[i, 1] = x1[i] * w
            detections[i, 2] = y1[i] * h
            detections[i, 3] = (x2[i] - x1[i]) * w
            detections[i, 4] = (y2[i] - y1[i]) * h

        detections[:, 5] = df["score"].values
        detections[:, 6] = df["label"].values

        coco_dt = coco_gt.loadRes(detections)
        img_ids_eval = list(set(detections[:, 0].astype(int)))
    else:
        coco_dt = coco_gt.loadRes(pred_path)
        img_ids_eval = coco_gt.getImgIds()

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = img_ids_eval
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats.tolist()


def iou_2d(boxes1, boxes2):
    """
    Vectorized IoU computation for arrays of boxes.
    boxes1, boxes2: numpy arrays of shape (N, 4)
    """
    boxes1 = np.array(boxes1, dtype=float)
    boxes2 = np.array(boxes2, dtype=float)

    # Compute intersection
    xA = np.maximum(boxes1[:, 0], boxes2[:, 0])
    yA = np.maximum(boxes1[:, 1], boxes2[:, 1])
    xB = np.minimum(boxes1[:, 2], boxes2[:, 2])
    yB = np.minimum(boxes1[:, 3], boxes2[:, 3])

    inter = np.clip(xB - xA, 0, None) * np.clip(yB - yA, 0, None)
    area1 = np.clip((boxes1[:, 2] - boxes1[:, 0]), 0, None) * np.clip((boxes1[:, 3] - boxes1[:, 1]), 0, None)
    area2 = np.clip((boxes2[:, 2] - boxes2[:, 0]), 0, None) * np.clip((boxes2[:, 3] - boxes2[:, 1]), 0, None)

    iou = inter / (area1 + area2 - inter + 1e-6)
    return iou

def compute_agreement(pred_files, gt_path):
    """
    Estimate inter-model agreement based on overlapping detections.
    Used as a soft causal regularizer for reliability priors.
    """
    #from ensemble_boxes_wbf.iou import iou_2d
    from itertools import combinations

    coco_gt = COCO(gt_path)
    agreements = []

    # Load all detections
    detections = []
    for path in pred_files:
        df = pd.read_csv(path)
        detections.append(df)

    for (i, detA), (j, detB) in combinations(enumerate(detections), 2):
        # Match by image_id and label, approximate consensus
        merged = pd.merge(detA, detB, on=["img_id", "label"], suffixes=("_a", "_b"))
        if len(merged) == 0:
            agreements.append(0.0)
            continue
        ious = iou_2d(
            merged[["x1_a", "y1_a", "x2_a", "y2_a"]].values,
            merged[["x1_b", "y1_b", "x2_b", "y2_b"]].values
        )
        agreements.append(np.mean(ious > 0.5))

    return np.mean(agreements)


def estimate_reliabilities(pred_files, gt_path, metric="mAP", use_agreement=True):
    """
    Estimate model reliabilities based on validation performance and (optionally) inter-model agreement.
    """
    print("\n🔧 Estimating detector reliabilities...")

    #perf_scores = [evaluate_model(f, gt_path, metric=metric) for f in pred_files]
    perf_scores = [evaluate_model(f, gt_path, metric=metric)[0] for f in pred_files]

    reliabilities = np.array(perf_scores, dtype=float)

    # Normalize between 0 and 1
    reliabilities = (reliabilities - reliabilities.min()) / (reliabilities.max() - reliabilities.min() + 1e-8)

    if use_agreement:
        agreement_factor = compute_agreement(pred_files, gt_path)
        reliabilities = 0.7 * reliabilities + 0.3 * agreement_factor  # softer, more causal scaling

    # Normalize relative to best model instead of sum
    reliabilities = np.clip(reliabilities, 0.05, 1.0)
    reliabilities /= np.max(reliabilities)
    reliabilities = np.clip(reliabilities, 0.6, 1.0)
    print("\n📊 Reliability priors:")
    for p, r in zip(pred_files, reliabilities):
        #print(f"  {os.path.basename(p):25s} -> {r:.3f}")
        #print(f"  {os.path.basename(p):25s} -> {float(np.mean(r)):.3f}")
        print(f"  {os.path.basename(p):25s} -> {float(r):.3f}")

    return reliabilities.tolist()


def get_or_estimate_reliabilities(pred_files, gt_path, cache_path="outputs/reliabilities.json",
                                  metric="mAP", use_agreement=True):
    """
    Load reliability priors if already cached, else compute and save them.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        print(f"🧩 Loading cached reliabilities from {cache_path}")
        with open(cache_path, "r") as f:
            reliabilities = json.load(f)
        if len(reliabilities) != len(pred_files):
            print("⚠️ Cached reliability size mismatch — recomputing.")
            reliabilities = estimate_reliabilities(pred_files, gt_path, metric, use_agreement)
    else:
        reliabilities = estimate_reliabilities(pred_files, gt_path, metric, use_agreement)
        with open(cache_path, "w") as f:
            json.dump(reliabilities, f, indent=2)

    return reliabilities
