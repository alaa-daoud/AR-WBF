"""
COCO Dataset Utilities
----------------------
Loads COCO annotations and converts COCO JSON predictions into
lists suitable for fusion functions.
"""

import json
from pycocotools.coco import COCO

def load_coco_annotations(annotations_file: str):
    """
    Loads COCO annotations.

    Args:
        annotations_file: path to COCO instances_*.json

    Returns:
        coco: COCO API instance
        imgs: list of image metadata dictionaries
    """
    coco = COCO(annotations_file)
    imgs = coco.loadImgs(coco.getImgIds())
    return coco, imgs

def coco_json_to_fusion_lists(json_files: list):
    """
    Converts a list of COCO-format JSON pred files (one per model)
    into fusion-ready lists.

    Args:
        json_files: list of file paths, each containing COCO bounding box predictions

    Returns:
        boxes_list, scores_list, labels_list
    """
    all_boxes, all_scores, all_labels = [], [], []
    for jf in json_files:
        preds = json.load(open(jf, "r"))
        boxes, scores, labels = [], [], []
        for pred in preds:
            # COCO bbox format: [x, y, width, height] -> convert to [x1, y1, x2, y2]
            x, y, w, h = pred["bbox"]
            boxes.append([x, y, x + w, y + h])
            scores.append(pred["score"])
            labels.append(pred["category_id"])
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
    return all_boxes, all_scores, all_labels
