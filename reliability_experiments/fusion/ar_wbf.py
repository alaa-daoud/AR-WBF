"""
Adaptive Reliability-Weighted Boxes Fusion (AR-WBF)
----------------------------------------------------
Centralized extension of Weighted Boxes Fusion (WBF)
Features:
  1. Per-model reliability weighting.
  2. Adaptive IoU threshold based on object size or category.
Author: Your Name
Inspired by: Solovyev et al., 2021 (Weighted Boxes Fusion)
Compatible with ensemble-boxes API.
"""

import numpy as np
from ensemble_boxes_wbf import (
    prefilter_boxes,
    get_weighted_box,
    find_matching_box_fast,
)


def weighted_boxes_fusion_adaptive(
    boxes_list,
    scores_list,
    labels_list,
    weights=None,
    iou_thr=0.55,
    skip_box_thr=0.0,
    reliability=None,
    adaptive_iou=False,
    image_shape=None,
    delta=0.15,
    conf_type="avg",
    allows_overflow=False
):
    if weights is None:
        weights = np.ones(len(boxes_list))
    else:
        weights = np.array(weights)

    if reliability is None:
        reliability = np.ones(len(boxes_list))
    else:
        reliability = np.array(reliability)

    # Build per-box weight matrix: weights * reliability
    fusion_weights = []
    for m in range(len(boxes_list)):
        per_box_weights = [weights[m] * reliability[m]] * len(boxes_list[m])
        fusion_weights.append(per_box_weights)

    filtered_boxes = prefilter_boxes(
        boxes_list,
        scores_list,
        labels_list,
        fusion_weights,
        skip_box_thr,
        is_weight_matrix=True,
        skip_checks=False
    )

    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    iou_thr_cat = dict()
    if adaptive_iou:
        image_area = image_shape[0] * image_shape[1] if image_shape else 1.0
        for label in filtered_boxes:
            cat_boxes = filtered_boxes[label][:, 4:]
            cat_areas = (cat_boxes[:, 2] - cat_boxes[:, 0]) * (cat_boxes[:, 3] - cat_boxes[:, 1])
            mean_area = np.mean(cat_areas) if len(cat_areas) > 0 else 0
            adaptive_thr = float(iou_thr + delta * np.log1p(mean_area / image_area))
            iou_thr_cat[label] = float(np.clip(adaptive_thr, 0.4, 0.9))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = np.empty((0, 8))

        for j in range(len(boxes)):
            match_iou = iou_thr_cat[label] if adaptive_iou else iou_thr
            index, best_iou = find_matching_box_fast(weighted_boxes, boxes[j], match_iou)

            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(np.array(new_boxes[index]), conf_type)
            else:
                new_boxes.append([boxes[j]])
                weighted_boxes = np.vstack((weighted_boxes, boxes[j].reshape(1, -1)))

        overall_boxes.extend(weighted_boxes)

    overall_boxes = np.array(overall_boxes)
    boxes = overall_boxes[:, 4:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0].astype(np.int32)
    return boxes, scores, labels


# ----------------------------------------
# 🩺 Local clone of WBF that is compatible
# ----------------------------------------
def manual_wbf_compatible(boxes_list, scores_list, labels_list, weights):
    skip_box_thr = 0.0
    iou_thr = 0.55
    conf_type = 'avg'

    # Always convert to weight matrix
    weight_matrix = []
    for m in range(len(boxes_list)):
        if isinstance(weights[m], (float, int)):
            weight_matrix.append([weights[m]] * len(boxes_list[m]))
        else:
            weight_matrix.append(weights[m])

    filtered_boxes = prefilter_boxes(
        boxes_list,
        scores_list,
        labels_list,
        weight_matrix,
        skip_box_thr,
        is_weight_matrix=True,
        skip_checks=False
    )

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = np.empty((0, 8))

        for j in range(len(boxes)):
            index, _ = find_matching_box_fast(weighted_boxes, boxes[j], iou_thr)
            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(
                    np.array(new_boxes[index]), conf_type
                )
            else:
                new_boxes.append([boxes[j]])
                weighted_boxes = np.vstack(
                    (weighted_boxes, boxes[j].reshape(1, -1))
                )

        overall_boxes.extend(weighted_boxes)

    overall_boxes = np.array(overall_boxes)
    return (
        overall_boxes[:, 4:],            # boxes
        overall_boxes[:, 1],             # scores
        overall_boxes[:, 0].astype(int)  # labels
    )


# --------------------------------
# 🧪 DEMO + UNIT TEST
# --------------------------------
def demo_comparison():
    boxes_list = [
        [[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.7, 0.7]],
        [[0.12, 0.12, 0.38, 0.38], [0.52, 0.52, 0.72, 0.72]],
    ]
    scores_list = [
        [0.9, 0.6],
        [0.8, 0.75],
    ]
    labels_list = [
        [1, 2],
        [1, 2],
    ]

    weights_matrix = [
        [1.0] * len(boxes_list[0]),
        [1.0] * len(boxes_list[1]),
    ]

    print("\n🔹 Standard WBF (baseline)")
    boxes_wbf, scores_wbf, labels_wbf = manual_wbf_compatible(
        boxes_list, scores_list, labels_list, weights=weights_matrix
    )
    for b, s, l in zip(boxes_wbf, scores_wbf, labels_wbf):
        print(f"Label {l} | Score {s:.2f} | Box {b}")

    print("\n🔸 AR-WBF with reliability=[0.9, 0.5], adaptive IoU=True")
    boxes_ar, scores_ar, labels_ar = weighted_boxes_fusion_adaptive(
        boxes_list, scores_list, labels_list,
        weights=[1.0, 1.0],
        reliability=[0.9, 0.5],
        adaptive_iou=True,
        image_shape=(1000, 1000),
        delta=0.3
    )
    for b, s, l in zip(boxes_ar, scores_ar, labels_ar):
        print(f"Label {l} | Score {s:.2f} | Box {b}")

    print("\n🔸 AR-WBF with reliability=[1.0, 0.1] (Model 1 trusted more)")
    boxes_ar1, scores_ar1, labels_ar1 = weighted_boxes_fusion_adaptive(
        boxes_list, scores_list, labels_list,
        weights=[1.0, 1.0],
        reliability=[1.0, 0.1],
        adaptive_iou=False
    )
    for b, s, l in zip(boxes_ar1, scores_ar1, labels_ar1):
        print(f"Label {l} | Score {s:.2f} | Box {b}")

    print("\n🔸 AR-WBF with reliability=[0.1, 1.0] (Model 2 trusted more)")
    boxes_ar2, scores_ar2, labels_ar2 = weighted_boxes_fusion_adaptive(
        boxes_list, scores_list, labels_list,
        weights=[1.0, 1.0],
        reliability=[0.1, 1.0],
        adaptive_iou=False
    )
    for b, s, l in zip(boxes_ar2, scores_ar2, labels_ar2):
        print(f"Label {l} | Score {s:.2f} | Box {b}")
def demo_comparison1():
    boxes_list = [
        [[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.7, 0.7]],
        [[0.12, 0.12, 0.38, 0.38], [0.52, 0.52, 0.72, 0.72]],
    ]
    scores_list = [
        [0.9, 0.6],
        [0.8, 0.75],
    ]
    labels_list = [
        [1, 2],
        [1, 2],
    ]

    weights_matrix = [
        [1.0] * len(boxes_list[0]),
        [1.0] * len(boxes_list[1]),
    ]

    boxes_wbf, scores_wbf, labels_wbf = manual_wbf_compatible(
        boxes_list, scores_list, labels_list, weights=weights_matrix
    )

    boxes_ar, scores_ar, labels_ar = weighted_boxes_fusion_adaptive(
        boxes_list, scores_list, labels_list,
        weights=[1.0, 1.0],
        reliability=[0.9, 0.5],
        adaptive_iou=True,
        image_shape=(1000, 1000),
        delta=0.3
    )

    print("Standard WBF boxes:", len(boxes_wbf))
    print("AR-WBF boxes:", len(boxes_ar))


def test_equivalence_to_wbf():
    boxes_list = [
        [[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.7, 0.7]],
        [[0.12, 0.12, 0.38, 0.38], [0.52, 0.52, 0.72, 0.72]],
    ]
    scores_list = [
        [0.9, 0.6],
        [0.8, 0.75],
    ]
    labels_list = [
        [1, 2],
        [1, 2],
    ]

    weights_matrix = [
        [1.0] * len(boxes_list[0]),
        [1.0] * len(boxes_list[1]),
    ]

    boxes_wbf, scores_wbf, labels_wbf = manual_wbf_compatible(
        boxes_list, scores_list, labels_list, weights=weights_matrix
    )

    boxes_ar, scores_ar, labels_ar = weighted_boxes_fusion_adaptive(
        boxes_list, scores_list, labels_list,
        weights=[1.0, 1.0],
        reliability=[1.0, 1.0],
        adaptive_iou=False
    )

    assert np.allclose(boxes_wbf, boxes_ar), "Boxes mismatch"
    assert np.allclose(scores_wbf, scores_ar), "Scores mismatch"
    assert np.all(labels_wbf == labels_ar), "Labels mismatch"
    print("✅ Test passed: AR-WBF is identical to WBF when extensions disabled")


if __name__ == "__main__":
    demo_comparison()
    test_equivalence_to_wbf()
