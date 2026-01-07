"""
Standard NMS Fusion
-------------------
Simple wrapper around torchvision.ops.nms to serve as a baseline fusion method.
"""

import torch
import torchvision

def fuse(boxes_list, scores_list, labels_list, iou_thr=0.5):
    """
    Apply traditional NMS on the union of all model predictions.

    Args:
        boxes_list: list of box lists (per model)
        scores_list: list of score lists
        labels_list: list of label lists
        iou_thr: IoU threshold for NMS

    Returns:
        nms_boxes, nms_scores, nms_labels
    """
    # flatten all lists
    boxes = torch.tensor([b for model in boxes_list for b in model], dtype=torch.float32)
    scores = torch.tensor([s for model in scores_list for s in model], dtype=torch.float32)
    labels = [l for model in labels_list for l in model]

    idxs = torchvision.ops.nms(boxes, scores, iou_thr)
    boxes_out = boxes[idxs].tolist()
    scores_out = scores[idxs].tolist()
    labels_out = [labels[i] for i in idxs.tolist()]
    return boxes_out, scores_out, labels_out
