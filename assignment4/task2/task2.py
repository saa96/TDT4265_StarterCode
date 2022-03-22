import numpy as np
import matplotlib.pyplot as plt
from sympy import re
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE
    a = prediction_box
    b = gt_box
    
    # Compute intersection
    dx = min(a[2],b[2]) - max(a[0],b[0])
    dy = min(a[3],b[3]) - max(a[1],b[1])

    if (dx >= 0) and (dy >= 0):
        intersect = dx*dy
    else:
        intersect = 0

    # Compute union
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])

    tot_area = (area_a + area_b) - intersect

    # Compute IOU
    iou = intersect/tot_area
    #print("Intersect: {}, tot area: {}, iou: {}".format(intersect, tot_area, iou))


    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if (num_tp + num_fp) == 0:
        return 1
    else:
        return num_tp/(num_tp + num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if (num_tp + num_fn) == 0:
        return 0
    else:
        return num_tp/(num_tp + num_fn)


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    possible_matches = []
    for i,p_box in enumerate(prediction_boxes):
        for j,gt_box in enumerate(gt_boxes):
            iou = calculate_iou(p_box,gt_box)
            if iou >= iou_threshold:
                possible_matches.append((i, j, iou))
                
    # Sort all matches on IoU in descending order       
    sorted_matches = sorted(possible_matches, key = lambda x: x[2],reverse=True)
    
    # Find all matches with the highest IoU threshold
    prediction_boxes_new = []
    gt_boxes_new = []
    while len(sorted_matches) > 0:
        p_box_index = sorted_matches[0][0]
        gt_box_index = sorted_matches[0][1]
        
        p_box = list(prediction_boxes[p_box_index])
        gt_box = list(gt_boxes[gt_box_index])
        
        prediction_boxes_new.append(p_box)
        gt_boxes_new.append(gt_box)
        
        sorted_matches = [(match[0], match[1], match[2]) for match in sorted_matches if match[0] != p_box_index and match[1] != gt_box_index]
    
    return np.array(prediction_boxes_new), np.array(gt_boxes_new)


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    p_matched, gt_matched = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    
    true_positive = p_matched.shape[0]
    false_positive = prediction_boxes.shape[0] - true_positive
    false_negative = gt_boxes.shape[0] - gt_matched.shape[0]
    
    return {"true_pos": true_positive, "false_pos": false_positive, "false_neg": false_negative}


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    
    total_true_pos = 0
    total_false_pos = 0
    total_false_neg = 0
    
    for prediction_boxes in all_prediction_boxes:
        for gt_boxes in all_gt_boxes:
        
            result = calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold)
        
            total_true_pos += result["true_pos"]
            total_false_pos += result["false_pos"]
            total_false_neg += result["false_neg"]
        
    total_precision = calculate_precision(total_true_pos,total_false_pos,total_false_neg)
    total_recall = calculate_recall(total_true_pos, total_false_pos, total_false_neg)
    
    return (total_precision,total_recall)


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    precisions = [] 
    recalls = []
    
    for threshold in confidence_thresholds:
        all_prediction_boxes_copy = all_prediction_boxes.copy()
        for i,predicted_boxes in enumerate(all_prediction_boxes_copy):
            score = confidence_scores[i]
            
            score_index = [j for j in range(len(score)) if score[j] > threshold]
            predicted_boxes = [predicted_boxes[j] for j in score_index]
            
            all_prediction_boxes_copy[i] = np.array(predicted_boxes)

        presicion,recall = calculate_precision_recall_all_images( all_prediction_boxes_copy, all_gt_boxes, iou_threshold)
        
        precisions.append(presicion)
        recalls.append(recall)
        
    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    p = np.zeros_like(recall_levels)
    
    for i,r_lvl in enumerate(recall_levels):
        r = [j for j in range(len(recalls)) if recalls[j] >= r_lvl]
        if len(r) > 0:
            p[i] = max(precisions[r])
        else:
            p[i] = 0
        print(f"r = {r}\n p = {p}")
        
    average_precision = np.sum(p)/p.shape[0]
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
