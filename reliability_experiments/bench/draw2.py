import sys
import os
import cv2
import numpy as np

# --- Ensure correct import paths ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ENSEMBLE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ENSEMBLE_PATH)

# --- Import fusion methods ---
from fusion.aar_wbf import weighted_boxes_fusion_adaptive
from ensemble_boxes_wbf_experimental import weighted_boxes_fusion_experimental as weighted_boxes_fusion



def draw_dotted_box(img, box_rel, color, thickness=3, gap=5):
    """Draw a dotted box (relative coords [0,1]) on the image."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(box_rel[0] * w), int(box_rel[1] * h), int(box_rel[2] * w), int(box_rel[3] * h)]

    def draw_dotted_line(pt1, pt2):
        dist = int(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
        for i in range(0, dist, gap * 2):
            start = (
                int(pt1[0] + (pt2[0] - pt1[0]) * i / dist),
                int(pt1[1] + (pt2[1] - pt1[1]) * i / dist),
            )
            end = (
                int(pt1[0] + (pt2[0] - pt1[0]) * min(i + gap, dist) / dist),
                int(pt1[1] + (pt2[1] - pt1[1]) * min(i + gap, dist) / dist),
            )
            cv2.line(img, start, end, color, thickness)

    draw_dotted_line((x1, y1), (x2, y1))
    draw_dotted_line((x2, y1), (x2, y2))
    draw_dotted_line((x2, y2), (x1, y2))
    draw_dotted_line((x1, y2), (x1, y1))


def put_label_with_background(img, text, position_rel, color, text_color=(255, 255, 255)):
    """Draw bold text with colored rectangular background using relative coords."""
    h, w = img.shape[:2]
    x = int(position_rel[0] * w)
    y = int(position_rel[1] * h)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Clamp inside image
    x = max(0, min(x, img.shape[1] - text_w - 4))
    y = max(text_h + 6, min(y, img.shape[0] - 2))

    # Background rectangle
    cv2.rectangle(img, (x, y - text_h - 6), (x + text_w + 6, y + 2), color, -1)
    # Bold white text
    cv2.putText(img, text, (x + 3, y - 5), font, font_scale, text_color, thickness + 1, cv2.LINE_AA)


def abs_box(box_rel, img_shape):
    """Convert relative [0,1] box to absolute pixel coordinates."""
    h, w = img_shape[:2]
    return [int(box_rel[0] * w), int(box_rel[1] * h), int(box_rel[2] * w), int(box_rel[3] * h)]


# -------------------------------------------------------------------
# Fusion + Visualization
# -------------------------------------------------------------------

def fuse_and_visualize(image_path, output_prefix="fusion_result", class_name="object"):
    # ---- Load image ----
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = img.shape[:2]

    # ---- Example input: relative coordinates (normalized [0,1]) ----
    #box_yolo = [0.23351562499999998,0.09662060889929742,0.7257921875,0.9871615925058549]
    #box_effdet = [0.5903951168060303,0.2523801030822325,0.8190227508544922,0.9729165427857874]
    box_yolo = [0.5602578125,0.5646454352441613,0.692759375,0.6967940552016985]
    box_effdet = [0.5609851360321045,0.5648363765384488,0.6994250774383545,0.6988362111863057]
    score_yolo = 0.90
    score_effdet = 0.82
    label = 24  # same class id

    # ---- Prepare for fusion (already relative) ----
    boxes_list = [[box_yolo], [box_effdet]]
    scores_list = [[score_yolo], [score_effdet]]
    labels_list = [[label], [label]]
    weights = [1.0, 1.0]
    reliabilities = [1.0, 0.6]  # Example reliability priors

    # ============================================================
    # 🔸 WBF
    # ============================================================
    boxes_wbf, scores_wbf, labels_wbf = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights,
        iou_thr=0.5,
        skip_box_thr=0.0,
        conf_type="avg",
        allows_overflow=False,
        skip_checks=False,
    )

    # ============================================================
    # 🔸 AR-WBF
    # ============================================================
    boxes_arwbf, scores_arwbf, labels_arwbf = weighted_boxes_fusion_adaptive(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights,
        reliability=reliabilities,
        adaptive_iou=True,
        iou_thr=0.5,
        skip_box_thr=0.0,
        conf_type="avg",
        allows_overflow=False,
    )

    # ============================================================
    # 🔸 Visualization
    # ============================================================
    img_wbf = img.copy()
    img_arwbf = img.copy()

    # --- Draw YOLO (red, dotted, top-left) ---
    draw_dotted_box(img_wbf, box_yolo, color=(0, 0, 255), thickness=2)
    put_label_with_background(img_wbf, f"{class_name} {score_yolo:.2f}", (box_yolo[0], box_yolo[1] - 0.01), color=(0, 0, 255))

    # --- Draw EfficientDet (blue, dotted, top-right) ---
    draw_dotted_box(img_wbf, box_effdet, color=(255, 0, 0), thickness=2)
    put_label_with_background(img_wbf, f"{class_name} {score_effdet:.2f}", (box_effdet[2] - 0.15, box_effdet[1] - 0.01), color=(255, 0, 0))

    # Copy to AR-WBF image too
    draw_dotted_box(img_arwbf, box_yolo, color=(0, 0, 255), thickness=2)
    put_label_with_background(img_arwbf, f"{class_name} {score_yolo:.2f}", (box_yolo[0], box_yolo[1] - 0.01), color=(0, 0, 255))
    draw_dotted_box(img_arwbf, box_effdet, color=(255, 0, 0), thickness=2)
    put_label_with_background(img_arwbf, f"{class_name} {score_effdet:.2f}", (box_effdet[2] - 0.15, box_effdet[1] - 0.01), color=(255, 0, 0))

    # --- Draw fused results (solid) ---
    for b_rel, s in zip(boxes_wbf, scores_wbf):
        b = abs_box(b_rel, img.shape)
        cv2.rectangle(img_wbf, (b[0], b[1]), (b[2], b[3]), (0, 165, 255), 2)  # Orange
        put_label_with_background(img_wbf, f"{class_name} {s:.2f}", ((b[0] / w) + 0.005, (b[3] / h) - 0.02), color=(0, 165, 255))

    for b_rel, s in zip(boxes_arwbf, scores_arwbf):
        b = abs_box(b_rel, img.shape)
        cv2.rectangle(img_arwbf, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)  # Green
        put_label_with_background(img_arwbf, f"{class_name} {s:.2f}", ((b[0] / w) + 0.005, (b[3] / h) - 0.02), color=(0, 255, 0))

    # ---- Save outputs ----
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    out_wbf = f"{output_prefix}_wbf.jpg"
    out_arwbf = f"{output_prefix}_arwbf.jpg"
    cv2.imwrite(out_wbf, img_wbf)
    cv2.imwrite(out_arwbf, img_arwbf)
    print(f"✅ Saved: {out_wbf}")
    print(f"✅ Saved: {out_arwbf}")


# -------------------------------------------------------------------
# Example run
# -------------------------------------------------------------------
if __name__ == "__main__":
    fuse_and_visualize("16010.jpg", output_prefix="results/comparison16010", class_name="zebra")