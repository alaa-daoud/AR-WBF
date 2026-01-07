"""
Input/Output Helpers
-------------------
Functions to convert model outputs to COCO JSON,
and save fusion results.
"""

import json

def save_to_coco_json(boxes_all, scores_all, labels_all, image_ids, out_file, image_shape=(1024, 1024)):
    out = []
    width, height = image_shape
    for boxes, scores, labels, img_id in zip(boxes_all, scores_all, labels_all, image_ids):
        for box, score, label in zip(boxes, scores, labels):
            x = float(box[0]) * width
            y = float(box[1]) * height
            w = float(box[2] - box[0]) * width
            h = float(box[3] - box[1]) * height
            out.append({
                "image_id": int(img_id),
                "category_id": int(label),
                "bbox": [x, y, w, h],
                "score": float(score),
            })

    import json
    with open(out_file, "w") as f:
        json.dump(out, f)
    return out_file
