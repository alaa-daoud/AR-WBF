"""
Evaluation Metrics using pycocotools
-----------------------------------
Wraps COCOeval for bounding box metrics.
"""

from pycocotools.cocoeval import COCOeval

def compute_coco_metrics(coco_gt, preds_json, iou_type="bbox"):
    """
    Compute COCO standard metrics.

    Args:
        coco_gt: COCO object of ground truth
        preds_json: JSON file with predictions
        iou_type: 'bbox'

    Returns:
        dict of metrics
    """
    coco_pred = coco_gt.loadRes(preds_json)
    evaluator = COCOeval(coco_gt, coco_pred, iou_type)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats
    return {
        "mAP@[.5:.95]": stats[0],
        "mAP@0.5": stats[1],
        "Recall@0.5": stats[8]
    }
