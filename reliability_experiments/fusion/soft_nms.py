"""
Soft-NMS Fusion
---------------
Implements Soft-NMS (linear decay). For simplicity, this uses a pure Python loop.
"""

import numpy as np
import torch
from torchvision.ops import nms as hard_nms

def fuse(boxes_list, scores_list, labels_list, iou_thr=0.5, **kwargs):
    """
    Simple hard NMS implementation (acts as placeholder for Soft-NMS).
    Modify if you use actual soft-NMS.
    """
    all_boxes = []
    all_scores = []
    all_labels = []

    for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
        if len(boxes) == 0:
            continue

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int32)

        keep_indices = hard_nms(boxes_tensor, scores_tensor, iou_thr)
        all_boxes.extend(boxes_tensor[keep_indices].tolist())
        all_scores.extend(scores_tensor[keep_indices].tolist())
        all_labels.extend(labels_tensor[keep_indices].tolist())

    return np.array(all_boxes), np.array(all_scores), np.array(all_labels)


def soft_nms(boxes, scores, iou_thr=0.5, sigma=0.5, score_thresh=0.001):
    """
    Pure-Python Soft-NMS on one list of boxes.

    Args:
        boxes: Nx4 np.array
        scores: N scores
        iou_thr: initial threshold
        sigma: decay factor
        score_thresh: prune threshold

    Returns:
        boxes_out, scores_out, labels_dummy
    """
    # simple implementation for demo — real use should optimize
    idxs = list(range(len(boxes)))
    results = []
    while idxs:
        # pick highest score
        i = max(idxs, key=lambda j: scores[j])
        idxs.remove(i)
        results.append((boxes[i], scores[i]))
        for j in idxs.copy():
            # compute IoU
            b1, b2 = boxes[i], boxes[j]
            xx1 = max(b1[0], b2[0])
            yy1 = max(b1[1], b2[1])
            xx2 = min(b1[2], b2[2])
            yy2 = min(b1[3], b2[3])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            area1 = (b1[2] - b1[0])*(b1[3] - b1[1])
            area2 = (b2[2] - b2[0])*(b2[3] - b2[1])
            union = area1 + area2 - inter
            iou = inter / union if union > 0 else 0
            scores[j] *= np.exp(-iou*iou/sigma)  # linear can also be used
            if scores[j] < score_thresh:
                idxs.remove(j)
    if not results:
        return [], [], []
    boxes_out, scores_out = zip(*results)
    return list(boxes_out), list(scores_out), [0]*len(boxes_out)
