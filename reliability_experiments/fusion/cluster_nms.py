"""
Cluster-NMS Baseline
--------------------
Clusters overlapping boxes and reduces them with mean coordinate.
"""

import numpy as np

def fuse(boxes_list, scores_list, labels_list, iou_thr=0.5):
    """
    A very simple version of cluster-based NMS.
    """
    all_boxes = [b for model in boxes_list for b in model]
    all_scores = [s for model in scores_list for s in model]
    all_labels = [l for model in labels_list for l in model]

    clusters = []
    taken = set()

    for i in range(len(all_boxes)):
        if i in taken:
            continue
        base = all_boxes[i]
        members = [i]
        for j in range(i+1, len(all_boxes)):
            if j in taken:
                continue
            # simple IoU
            b1, b2 = np.array(base), np.array(all_boxes[j])
            xx1 = max(b1[0], b2[0])
            yy1 = max(b1[1], b2[1])
            xx2 = min(b1[2], b2[2])
            yy2 = min(b1[3], b2[3])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w*h
            union = ((b1[2]-b1[0])*(b1[3]-b1[1]) +
                     (b2[2]-b2[0])*(b2[3]-b2[1]) - inter)
            iou = inter/union if union > 0 else 0
            if iou > iou_thr:
                members.append(j)
                taken.add(j)
        cluster_boxes = np.array([all_boxes[m] for m in members])
        cluster_scores = np.array([all_scores[m] for m in members])
        # mean coords
        fused_box = cluster_boxes.mean(axis=0).tolist()
        fused_score = float(cluster_scores.mean())
        fused_label = all_labels[members[0]]
        clusters.append((fused_box, fused_score, fused_label))

    if not clusters:
        return [], [], []
    boxes_out, scores_out, labels_out = zip(*clusters)
    return list(boxes_out), list(scores_out), list(labels_out)
