import numpy as np
import os
import torch
import ujson as json

from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_predictions_preprocess(
        predicted_instances,
        min_allowed_score=0.0,
        is_odd=False):
    predicted_boxes, predicted_cls_probs, predicted_covar_mats = defaultdict(
        torch.Tensor), defaultdict(
        torch.Tensor), defaultdict(
        torch.Tensor)
    for predicted_instance in predicted_instances:
        # Remove predictions with undefined category_id. This is used when the training and
        # inference datasets come from different data such as COCO-->VOC or COCO-->OpenImages.
        # Only happens if not ODD dataset, else all detections will be removed.
        if len(predicted_instance['cls_prob']) == 81:
            cls_prob = predicted_instance['cls_prob'][:-1]
        else:
            cls_prob = predicted_instance['cls_prob']
        if not is_odd:
            skip_test = (
                                predicted_instance['category_id'] == -
                        1) or (
                                np.array(cls_prob).max(0) < min_allowed_score)
        else:
            skip_test = np.array(cls_prob).max(0) < min_allowed_score

        if skip_test:
            continue

        box_inds = predicted_instance['bbox']
        box_inds = np.array([box_inds[0],
                             box_inds[1],
                             box_inds[0] + box_inds[2],
                             box_inds[1] + box_inds[3]])

        predicted_boxes[predicted_instance['image_id']] = torch.cat((predicted_boxes[predicted_instance['image_id']].to(
            device), torch.as_tensor([box_inds], dtype=torch.float32).to(device)))

        predicted_cls_probs[predicted_instance['image_id']] = torch.cat(
            (predicted_cls_probs[predicted_instance['image_id']].to(
                device), torch.as_tensor([predicted_instance['cls_prob']], dtype=torch.float32).to(device)))

        box_covar = np.array(predicted_instance['bbox_covar'])
        transformation_mat = np.array([[1.0, 0, 0, 0],
                                       [0, 1.0, 0, 0],
                                       [1.0, 0, 1.0, 0],
                                       [0, 1.0, 0.0, 1.0]])

        cov_pred = np.matmul(
            np.matmul(
                transformation_mat,
                box_covar),
            transformation_mat.T).tolist()

        predicted_covar_mats[predicted_instance['image_id']] = torch.cat(
            (predicted_covar_mats[predicted_instance['image_id']].to(device),
             torch.as_tensor([cov_pred], dtype=torch.float32).to(device)))

    return dict({'predicted_boxes': predicted_boxes,
                 'predicted_cls_probs': predicted_cls_probs,
                 'predicted_covar_mats': predicted_covar_mats})


def get_per_frame_preprocessed_instances(
        cfg, inference_output_dir, min_allowed_score=0.0):
    prediction_file_name = os.path.join(inference_output_dir, 'results.json')

    print("Began pre-processing predicted instances...")
    try:
        preprocessed_predicted_instances = torch.load(
            os.path.join(
                inference_output_dir,
                "preprocessed_predicted_instances_{}.pth".format(min_allowed_score)),
            map_location=device)
    # Process predictions
    except FileNotFoundError:
        predicted_instances = json.load(open(prediction_file_name, 'r'))
        preprocessed_predicted_instances = eval_predictions_preprocess(
            predicted_instances, min_allowed_score)
        torch.save(
            preprocessed_predicted_instances,
            os.path.join(
                inference_output_dir,
                "preprocessed_predicted_instances_{}.pth".format(min_allowed_score)))
    print("Done!")

    return preprocessed_predicted_instances, None
