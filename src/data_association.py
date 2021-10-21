import os
import sys
import json
import core
from collections import defaultdict

# This is very ugly. Essential for now but should be fixed.
sys.path.append(os.path.join(core.top_dir(), 'src', 'detr'))

import torch
import numpy as np

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.engine import launch

# Project imports
from core.setup import setup_config, setup_arg_parser
from detectron2.structures import Boxes, pairwise_iou

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # Path to Ground truth and predicted instances
    path_gt = args.image_dir
    path_instances = os.path.join(args.output_dir, 'preprocessed_predicted_instances_0.3.pth')
    # Load predicted instances
    predicted_instances = torch.load(path_instances)

    # Dictionaries to store new data
    predicted_boxes, predicted_cls_probs, predicted_covar_mats, instance_key, cam_pose, = defaultdict(
        torch.Tensor), defaultdict(torch.Tensor), defaultdict(
        torch.Tensor), defaultdict(torch.Tensor), defaultdict(
        torch.Tensor)

    print('Starting IoU matching...')
    for image_id in predicted_instances['predicted_boxes']:
        # Open Ground truth JSON file
        json_name = '.'.join(image_id.split('.')[:2]) + '.json'
        file_path = os.path.join(path_gt, json_name)
        # Continue to next file if there is no ground truth
        try:
            with open(file_path) as f:
                gt_bbox = json.load(f)
        except FileNotFoundError:
            print('Image {} does not have ground truth'.format(image_id))
            continue

        # Ground truth bounding boxes
        gt_box_mean = np.array(gt_bbox['bbox'], dtype=np.float32)
        gt_object_key = np.array(gt_bbox['object_key'], dtype=np.float32)
        gt_pose = np.array(gt_bbox['pose'], dtype=np.float32)

        # Predicted instance data
        predicted_box_means = predicted_instances['predicted_boxes'][image_id].cpu().numpy()
        predicted_box_covariances = predicted_instances['predicted_covar_mats'][image_id].cpu().numpy()
        predicted_clas_probs = predicted_instances['predicted_cls_probs'][image_id].cpu().numpy()

        # Compute iou between gt boxes and all predicted boxes in frame
        frame_gt_boxes = Boxes(gt_box_mean)
        frame_predicted_boxes = Boxes(predicted_box_means)

        match_iou = pairwise_iou(frame_gt_boxes, frame_predicted_boxes)

        # Match GT and predicted boxes
        matched_instance = torch.argmax(match_iou, dim=1)

        # Loop over instances
        for gt_idx, match in enumerate(matched_instance):
            if match_iou[gt_idx][match] > 0:
                # Storing bbox
                predicted_boxes[image_id] = torch.cat((predicted_boxes[image_id].to(device),
                                                       torch.as_tensor([predicted_box_means[match]],
                                                                       dtype=torch.float32).to(device)))
                # Storing class probabilities
                predicted_cls_probs[image_id] = torch.cat((predicted_cls_probs[image_id].to(device),
                                                           torch.as_tensor([predicted_clas_probs[match]],
                                                                           dtype=torch.float32).to(device)))
                # Storing covariance matrices
                predicted_covar_mats[image_id] = torch.cat((predicted_covar_mats[image_id].to(device),
                                                            torch.as_tensor([predicted_box_covariances[match]],
                                                                            dtype=torch.float32).to(device)))
                # Storing tracking key
                instance_key[image_id] = torch.cat((instance_key[image_id].to(device),
                                                    torch.as_tensor([gt_object_key[gt_idx]],
                                                                    dtype=torch.float32).to(device)))
                # Storing tracking key
                cam_pose[image_id] = torch.cat((cam_pose[image_id].to(device),
                                                torch.as_tensor([gt_pose],
                                                                dtype=torch.float32).to(device)))

    processed_instances = dict({'predicted_boxes': predicted_boxes,
                                'predicted_cls_probs': predicted_cls_probs,
                                'predicted_covar_mats': predicted_covar_mats,
                                'predicted_instance_key': instance_key,
                                'camera_pose': cam_pose})

    torch.save(processed_instances, os.path.join(args.output_dir, "preprocessed_predicted_instances_key.pth"))
    print('Done!')


if __name__ == "__main__":
    # Create arg parser
    arg_parser = setup_arg_parser()

    args = arg_parser.parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
