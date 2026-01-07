import numpy as np
from ensemble_boxes_wbf import weighted_boxes_fusion_per_box_weights as wbf_per_box
from ensemble_boxes_wbf import prefilter_boxes

def iou(box1, box2):
    """Compute IoU (intersection over union) between two boxes."""
    xA = np.maximum(box1[0], box2[0])
    yA = np.maximum(box1[1], box2[1])
    xB = np.minimum(box1[2], box2[2])
    yB = np.minimum(box1[3], box2[3])

    inter_area = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)
    box1_area = np.maximum(box1[2] - box1[0], 0) * np.maximum(box1[3] - box1[1], 0)
    box2_area = np.maximum(box2[2] - box2[0], 0) * np.maximum(box2[3] - box2[1], 0)

    union = box1_area + box2_area - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def get_weighted_box(cluster_boxes, cluster_scores, cluster_weights, conf_type="avg", allows_overflow=False):
    """
    Compute a weighted average bounding box and confidence for a cluster of overlapping boxes.
    Args:
        cluster_boxes: np.array of shape (N, 4)
        cluster_scores: np.array of shape (N,)
        cluster_weights: np.array of shape (N,)
        conf_type: 'avg', 'max', or 'sum' – how to combine confidence scores
        allows_overflow: (unused, kept for compatibility)
    Returns:
        fused_box (np.array of shape (4,)), fused_score (float)
    """
    if len(cluster_boxes) == 0:
        return np.zeros(4), 0.0

    # Weighted average of coordinates
    total_weight = np.sum(cluster_scores * cluster_weights)
    if total_weight == 0:
        fused_box = np.mean(cluster_boxes, axis=0)
    else:
        fused_box = np.sum(cluster_boxes * (cluster_scores * cluster_weights)[:, None], axis=0) / total_weight

    # Confidence fusion
    if conf_type == "avg":
        fused_score = np.average(cluster_scores, weights=cluster_weights)
    elif conf_type == "max":
        fused_score = np.max(cluster_scores)
    elif conf_type == "sum":
        fused_score = np.sum(cluster_scores * cluster_weights) / np.sum(cluster_weights)
    else:
        fused_score = np.mean(cluster_scores)

    return fused_box, float(fused_score)


def weighted_boxes_fusion_adaptive1(
    boxes_list,
    scores_list,
    labels_list,
    weights=None,
    iou_thr=0.55,
    skip_box_thr=0.0,
    conf_type='avg',
    allows_overflow=False,
    reliability=None,
    adaptive_iou=False,
    image_shape=None,
    delta=0.2,
    alpha=0.15,
    base_iou=0.55
):
    """
    Adaptive Reliability-Weighted Box Fusion (AR-WBF)
    - Supports per-box dynamic weights based on model reliability and box characteristics.
    - Fully compatible with ensemble_boxes_wbf's per-box weight variant.

    Parameters:
    - boxes_list: list of list of boxes for each model
    - scores_list: list of list of confidence scores for each model
    - labels_list: list of list of labels for each model
    - weights: base weight per model
    - reliability: reliability score per model
    - adaptive_iou: if True, modulates weights based on box size
    - image_shape: [H, W] for box size normalization
    - delta: controls box size sigmoid scaling
    """
    if reliability is None:
        reliability = np.ones(len(boxes_list))
    else:
        reliability = np.array(reliability, dtype=float)
        # Normalize so that max = 1, avoids global score shrinkage
        if reliability.max() > 0:
            reliability /= reliability.max()

    # If no explicit weights, use ones
    if weights is None:
        weights = np.ones(len(boxes_list))
    else:
        weights = np.array(weights, dtype=float)

    # Effective per-model weights (trust-modulated)
    eff_weights = weights * reliability


    print(f">> DEBUG: len(weights) = {len(weights)}, len(boxes_list) = {len(boxes_list)}")
    print(f">> DEBUG: len(reliability) = {len(reliability)}, len(boxes_list) = {len(boxes_list)}")
    assert len(weights) == len(boxes_list)
    assert len(reliability) == len(boxes_list)

    per_box_weights_list = []
    for m in range(len(boxes_list)):
        model_weights = []
        for box in boxes_list[m]:
            weight = eff_weights[m]
            if adaptive_iou and image_shape is not None:
                width = box[2] - box[0]
                height = box[3] - box[1]
                box_size = width * height / (image_shape[0] * image_shape[1])
                scale = 1.0 / (1.0 + np.exp(- (box_size - delta)))
                weight *= scale
            model_weights.append(weight)
        per_box_weights_list.append(model_weights)

    # Ensure label format is consistent
    labels_list = [np.atleast_1d(l) for l in labels_list]

    boxes, scores, labels = wbf_per_box(
        boxes_list=boxes_list,
        scores_list=scores_list,
        labels_list=labels_list,
        weights=per_box_weights_list,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
        conf_type=conf_type,
        allows_overflow=allows_overflow,
    )

    return boxes, scores, labels

def weighted_boxes_fusion_adaptive(
    boxes_list,
    scores_list,
    labels_list,
    weights=None,
    iou_thr=0.5,
    skip_box_thr=0.00,
    conf_type='avg',
    allows_overflow=False,
    reliability=None,
    adaptive_iou=False,
    image_shape=None,
    delta=0.2,
    alpha=0.15,
    base_iou=0.55
):
    """
    Adaptive Reliability-Weighted Boxes Fusion (AR-WBF).
    Extends WBF with:
    1. Reliability priors (r_k): per-model trust weighting (causal prior)
    2. Adaptive IoU threshold τ(c, A): context-aware fusion (causal intervention)
    """

    # --- Normalize reliability priors ---
    if reliability is None:
        reliability = np.ones(len(boxes_list))
    else:
        reliability = np.array(reliability, dtype=float)
        if reliability.max() > 0:
            reliability /= reliability.max()

    if weights is None:
        weights = np.ones(len(boxes_list))
    else:
        weights = np.array(weights, dtype=float)

    #eff_weights = weights * reliability
    eff_weights = np.copy(weights)


    # --- Prefilter ---
    #boxes, scores, labels, weights = prefilter_boxes(
    #    boxes_list, scores_list, labels_list, eff_weights, skip_box_thr, skip_checks=False
    #)
    # Prefilter boxes by label using confidence and weights
    new_boxes = prefilter_boxes(
        boxes_list, scores_list, labels_list, eff_weights, skip_box_thr, skip_checks=False
    )

    # Initialize containers for final aggregated boxes, scores, and labels
    boxes, scores, labels, weights = [], [], [], []

    # Flatten all class-specific entries into global arrays
    for label in new_boxes:
        label_boxes = new_boxes[label]
        boxes.extend(label_boxes[:, 4:])      # x1, y1, x2, y2
        scores.extend(label_boxes[:, 1])      # score
        weights.extend(label_boxes[:, 2])     # model weight
        labels.extend(label_boxes[:, 0])      # class label

    # Convert to numpy arrays for later fusion steps
    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)
    weights = np.array(weights)



    if len(boxes) == 0:
        return [], [], []

    boxes_by_label = {}
    for i, label in enumerate(labels):
        boxes_by_label.setdefault(label, []).append((boxes[i], scores[i], weights[i]))

    fused_boxes, fused_scores, fused_labels = [], [], []

    for label, entries in boxes_by_label.items():
        boxes_arr = np.array([e[0] for e in entries])
        scores_arr = np.array([e[1] for e in entries])
        w_arr = np.array([e[2] for e in entries])

        used = np.zeros(len(boxes_arr), dtype=bool)

        # --- Adaptive IoU threshold ---
        if adaptive_iou:
            areas = (boxes_arr[:, 2] - boxes_arr[:, 0]) * (boxes_arr[:, 3] - boxes_arr[:, 1])
            mean_area = np.mean(areas)
            # Normalize roughly relative to average object size
            ref_area = np.mean(areas) + 1e-6
            scale = np.log1p(mean_area / ref_area)
            # Adaptive IoU varies mildly around the base value
            tau = base_iou * (1.0 + 0.1 * scale)
            tau = np.clip(tau, 0.45, 0.65)
        else:
            tau = base_iou

        # --- Fusion per cluster ---
        for i in range(len(boxes_arr)):
            if used[i]:
                continue
            used[i] = True
            cluster_boxes = [boxes_arr[i]]
            cluster_scores = [scores_arr[i]]
            cluster_weights = [w_arr[i]]

            for j in range(i + 1, len(boxes_arr)):
                if used[j]:
                    continue
                if iou(boxes_arr[i], boxes_arr[j]) > tau:
                    used[j] = True
                    cluster_boxes.append(boxes_arr[j])
                    cluster_scores.append(scores_arr[j])
                    cluster_weights.append(w_arr[j])

            cluster_boxes = np.array(cluster_boxes)
            cluster_scores = np.array(cluster_scores)
            cluster_weights = np.array(cluster_weights)

            fused_box, fused_score = get_weighted_box(
                cluster_boxes, cluster_scores, cluster_weights, conf_type, allows_overflow
            )
            reliability_factor = np.mean([
                reliability[min(len(reliability) - 1, int(m))] for m in cluster_weights / (np.max(cluster_weights) + 1e-6)
            ])
            fused_score *= 0.9 + 0.1 * reliability_factor  # smooth scaling to avoid harsh suppression

            fused_boxes.append(fused_box)
            fused_scores.append(fused_score)
            fused_labels.append(label)

    fused_boxes = np.array(fused_boxes)
    fused_scores = np.array(fused_scores)
    fused_labels = np.array(fused_labels)
    order = np.argsort(-fused_scores)
    return fused_boxes[order].tolist(), fused_scores[order].tolist(), fused_labels[order].tolist()

def weighted_boxes_fusion_adaptive_old(
    boxes_list,
    scores_list,
    labels_list,
    weights=None,
    iou_thr=0.55,
    skip_box_thr=0.0,
    conf_type='avg',
    allows_overflow=False,
    reliability=None,
    adaptive_iou=False,
    image_shape=None,
    delta=0.2
):
    """
    Adaptive Reliability-Weighted Box Fusion (AR-WBF)
    - Supports per-box dynamic weights based on model reliability and box characteristics.
    - Fully compatible with ensemble_boxes_wbf's per-box weight variant.

    Parameters:
    - boxes_list: list of list of boxes for each model
    - scores_list: list of list of confidence scores for each model
    - labels_list: list of list of labels for each model
    - weights: base weight per model
    - reliability: reliability score per model
    - adaptive_iou: if True, modulates weights based on box size
    - image_shape: [H, W] for box size normalization
    - delta: controls box size sigmoid scaling
    """
    if weights is None:
        weights = [1.0] * len(boxes_list)
    if reliability is None:
        reliability = [1.0] * len(boxes_list)

    print(f">> DEBUG: len(weights) = {len(weights)}, len(boxes_list) = {len(boxes_list)}")
    print(f">> DEBUG: len(reliability) = {len(reliability)}, len(boxes_list) = {len(boxes_list)}")
    assert len(weights) == len(boxes_list)
    assert len(reliability) == len(boxes_list)

    per_box_weights_list = []
    for m in range(len(boxes_list)):
        model_weights = []
        for box in boxes_list[m]:
            weight = weights[m] * reliability[m]
            if adaptive_iou and image_shape is not None:
                width = box[2] - box[0]
                height = box[3] - box[1]
                box_size = width * height / (image_shape[0] * image_shape[1])
                scale = 1.0 / (1.0 + np.exp(- (box_size - delta)))
                weight *= scale
            model_weights.append(weight)
        per_box_weights_list.append(model_weights)

    # Ensure label format is consistent
    labels_list = [np.atleast_1d(l) for l in labels_list]

    boxes, scores, labels = wbf_per_box(
        boxes_list=boxes_list,
        scores_list=scores_list,
        labels_list=labels_list,
        weights=per_box_weights_list,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
        conf_type=conf_type,
        allows_overflow=allows_overflow,
    )

    return boxes, scores, labels


