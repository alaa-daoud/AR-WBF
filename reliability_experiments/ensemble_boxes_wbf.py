# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


import warnings
import numpy as np

def get_each_vs_each(arr, func):
    x = len(arr)
    arr_tile = np.tile(arr, x)
    arr_repeat = np.repeat(arr, x)
    func_arr = func(arr_tile, arr_repeat)
    res = func_arr.reshape((x, x))
    return res

def get_iou_matrix(boxes):
    xA = get_each_vs_each(boxes[:, 0], np.maximum)
    yA = get_each_vs_each(boxes[:, 1], np.maximum)
    xB = get_each_vs_each(boxes[:, 2], np.minimum)
    yB = get_each_vs_each(boxes[:, 3], np.minimum)
    interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)

    # compute sum of areas each vs each
    boxArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sumArea = get_each_vs_each(boxArea, np.add)

    iou_matrix = interArea / (sumArea - interArea)
    return iou_matrix

def weighted_boxes_fusion_per_box_weights(
    boxes_list,
    scores_list,
    labels_list,
    weights=None,  # Should be list of weights (one per model) or list of weights per box
    iou_thr=0.55,
    skip_box_thr=0.0,
    conf_type='avg',
    allows_overflow=False,
    skip_checks=False,
):
    if weights is None:
        weights = [1.0] * len(boxes_list)

    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights. Resetting to 1.')
        weights = [1.0] * len(boxes_list)

    filtered_boxes = prefilter_boxes(
        boxes_list,
        scores_list,
        labels_list,
        weights,
        skip_box_thr,
        skip_checks,
        is_weight_matrix=True
    )

    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = []

        iou_matrix = get_iou_matrix(boxes[:, 4:])

        used_locations = set()
        for j in range(len(boxes)):
            if j in used_locations:
                continue
            locations = np.where(iou_matrix[j] > iou_thr)[0]
            set_loc = set(locations)
            locations = list(set_loc - used_locations)
            bs = boxes[locations]

            if conf_type == 'avg':
                new_boxes.append(len(bs))
            elif conf_type in ['box_and_model_avg', 'absent_model_aware_avg']:
                new_boxes.append(bs)

            wb = get_weighted_box(bs, conf_type)
            weighted_boxes.append(wb)
            used_locations |= set_loc

        if len(weighted_boxes) == 0:
            continue

        weighted_boxes = np.stack(weighted_boxes, axis=0)

        for i in range(len(weighted_boxes)):
            clustered_boxes = new_boxes[i]

            if conf_type == 'box_and_model_avg':
                clustered_boxes = np.array(clustered_boxes)
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weighted_boxes[i, 2]

                _, idx = np.unique(clustered_boxes[:, 3], return_index=True)
                cluster_weights = clustered_boxes[:, 2]
                unique_model_weights = clustered_boxes[idx, 2]

                total_weight = cluster_weights.sum()
                if total_weight > 0:
                    weighted_boxes[i, 1] *= unique_model_weights.sum() / total_weight
                else:
                    weighted_boxes[i, 1] = 0

            elif conf_type == 'absent_model_aware_avg':
                clustered_boxes = np.array(clustered_boxes)
                models = np.unique(clustered_boxes[:, 3]).astype(int)
                mask = np.ones(len(weights), dtype=bool)
                mask[models] = False

                total_weight = weighted_boxes[i, 2]
                absent_weight = sum([w for idx, w in enumerate(weights) if mask[idx]])
                if total_weight + absent_weight > 0:
                    weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / (total_weight + absent_weight)
                else:
                    weighted_boxes[i, 1] = 0

            elif conf_type == 'max':
                flat_weights = [w for model_weights in weights for w in (model_weights if isinstance(model_weights, list) else [model_weights])]
                weighted_boxes[i, 1] /= max(flat_weights)

            elif not allows_overflow:
                total_weight = sum([box[2] for box in clustered_boxes]) if isinstance(clustered_boxes, (list, np.ndarray)) else 1
                weighted_boxes[i, 1] *= min(len(weights), total_weight) / total_weight
            else:
                total_weight = sum([box[2] for box in clustered_boxes]) if isinstance(clustered_boxes, (list, np.ndarray)) else 1
                weighted_boxes[i, 1] *= total_weight / total_weight

        overall_boxes.append(weighted_boxes)

    if not overall_boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 4:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]

    return boxes, scores, labels

def prefilter_boxes(
    boxes_list,
    scores_list,
    labels_list,
    weights,
    skip_box_thr,
    skip_checks,
    is_weight_matrix=False
):
    new_boxes = {}

    for model_idx, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
        if len(boxes) == 0:
            continue

        if not skip_checks:
            assert len(boxes) == len(scores) == len(labels)

        for i in range(len(boxes)):
            score = scores[i]
            if score < skip_box_thr:
                continue

            label = int(labels[i])
            box = boxes[i]
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                continue

            # Per-box or per-model weight
            weight = weights[model_idx][i] if is_weight_matrix else weights[model_idx]

            if label not in new_boxes:
                new_boxes[label] = []

            new_boxes[label].append([
                label, score, weight, model_idx, x1, y1, x2, y2
            ])

    for label in new_boxes:
        new_boxes[label] = np.array(new_boxes[label])

    return new_boxes



def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box (label, score, weight, model index, x1, y1, x2, y2)
    """

    box = np.zeros(8, dtype=np.float32)
    conf = 0
    conf_list = []
    w = 0
    for b in boxes:
        box[4:] += (b[1] * b[4:])
        conf += b[1]
        conf_list.append(b[1])
        w += b[2]
    box[0] = boxes[0][0]
    if conf_type in ('avg', 'box_and_model_avg', 'absent_model_aware_avg'):
        total_weight = boxes[:, 2].sum()
        if total_weight == 0:
            # Fallback: treat as uniform average to avoid NaN/Inf
            box[1] = 0.0
        else:
            box[1] = conf / total_weight
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    box[2] = w
    box[3] = -1 # model index field is retained for consistency but is not used.
    box[4:] /= conf
    return box


def find_matching_box_fast(boxes_list, new_box, match_iou):
    """
        Reimplementation of find_matching_box with numpy instead of loops. Gives significant speed up for larger arrays
        (~100x). This was previously the bottleneck since the function is called for every entry in the array.
    """
    def bb_iou_array(boxes, new_box):
        # bb interesection over union
        xA = np.maximum(boxes[:, 0], new_box[0])
        yA = np.maximum(boxes[:, 1], new_box[1])
        xB = np.minimum(boxes[:, 2], new_box[2])
        yB = np.minimum(boxes[:, 3], new_box[3])

        interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxBArea = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])

        iou = interArea / (boxAArea + boxBArea - interArea)

        return iou

    if boxes_list.shape[0] == 0:
        return -1, match_iou

    # boxes = np.array(boxes_list)
    boxes = boxes_list

    ious = bb_iou_array(boxes[:, 4:], new_box[4:])

    ious[boxes[:, 0] != new_box[0]] = -1

    best_idx = np.argmax(ious)
    best_iou = ious[best_idx]

    if best_iou <= match_iou:
        best_iou = match_iou
        best_idx = -1

    return best_idx, best_iou


def weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=None,
        iou_thr=0.55,
        skip_box_thr=0.0,
        conf_type='avg',
        allows_overflow=False
):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes.
        'avg': average value,
        'max': maximum value,
        'box_and_model_avg': box and model wise hybrid weighted average,
        'absent_model_aware_avg': weighted average that takes into account the absent model.
    :param allows_overflow: false if we want confidence score not exceed 1.0

    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
        print('Unknown conf_type: {}. Must be "avg", "max" or "box_and_model_avg", or "absent_model_aware_avg"'.format(conf_type))
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr,
    is_weight_matrix=True)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = np.empty((0, 8))

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box_fast(weighted_boxes, boxes[j], iou_thr)

            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes = np.vstack((weighted_boxes, boxes[j].copy()))

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            clustered_boxes = new_boxes[i]
            if conf_type == 'box_and_model_avg':
                clustered_boxes = np.array(clustered_boxes)
                # weighted average for boxes
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weighted_boxes[i, 2]
                # identify unique model index by model index column
                _, idx = np.unique(clustered_boxes[:, 3], return_index=True)
                # rescale by unique model weights
                weighted_boxes[i, 1] = weighted_boxes[i, 1] *  clustered_boxes[idx, 2].sum() / weights.sum()
            elif conf_type == 'absent_model_aware_avg':
                clustered_boxes = np.array(clustered_boxes)
                # get unique model index in the cluster
                models = np.unique(clustered_boxes[:, 3]).astype(int)
                # create a mask to get unused model weights
                mask = np.ones(len(weights), dtype=bool)
                mask[models] = False
                # absent model aware weighted average
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / (weighted_boxes[i, 2] + weights[mask].sum())
            elif conf_type == 'max':
                weighted_boxes[i, 1] = weighted_boxes[i, 1] / weights.max()
            elif not allows_overflow:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * min(len(weights), len(clustered_boxes)) / weights.sum()
            else:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weights.sum()
        overall_boxes.append(weighted_boxes)
    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 4:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels
