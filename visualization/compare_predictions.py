# This method only compares DETR EN vs DETR NLL

import cv2
import numpy as np
import os
import ujson as json
import core
import sys

from scipy.stats import entropy
from matplotlib import cm
import torch

# This is very ugly. Essential for now but should be fixed.
sys.path.append(os.path.join(core.top_dir(), 'src', 'detr'))

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.engine import launch

# Project imports
from core.setup import setup_config, setup_arg_parser
from core.evaluation_tools import evaluation_utils_inference
from core.visualization_tools.probabilistic_visualizer import ProbabilisticVisualizer


# noinspection PyTypeChecker
def main(args, cfg=None):
    # Setup config
    if cfg is None:
        cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)

    cfg.defrost()
    cfg.ACTUAL_TEST_DATASET = args.test_dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Update value min_allowed_score and inference_output_dir
    inference_output_dir = args.output_dir
    import sys

    # Load DETR EN instances
    instances_en = torch.load(os.path.join(inference_output_dir, "detr_en.pth"),
                              map_location=device)
    # Load DETR NLL instances
    instances_nll = torch.load(os.path.join(inference_output_dir, "detr_nll.pth"),
                               map_location=device)

    # get metacatalog and image infos
    meta_catalog = MetadataCatalog.get(args.test_dataset)
    # Read all images and sort them
    image_folder = os.path.expanduser(args.image_dir)
    images_list = os.listdir(image_folder)
    images_list.sort()

    # Loop over all images and visualize errors
    for image_id in images_list:
        # Make sure image_id is valid
        split_id = image_id.split('.')
        split_id[1] = split_id[1] + '0' if len(split_id[1]) < 6 else split_id[1]
        image_id = '.'.join(split_id)
        # Read new image
        image = cv2.imread(
            os.path.join(
                args.image_dir,
                image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        v = ProbabilisticVisualizer(
            image,
            meta_catalog,
            scale=1.5)
        class_list = v.metadata.as_dict()['thing_classes']

        # Extract metadata DETR EN
        predicted_box_means_en = instances_en['predicted_boxes'][image_id].cpu().numpy()
        predicted_box_covariances_en = instances_en['predicted_covar_mats'][image_id].cpu().numpy()
        predicted_cls_probs_en = instances_en['predicted_cls_probs'][image_id]
        # Extract metadata DETR NLL
        predicted_box_means_nll = instances_nll['predicted_boxes'][image_id].cpu().numpy()
        predicted_box_covariances_nll = instances_nll['predicted_covar_mats'][image_id].cpu().numpy()
        predicted_cls_probs_nll = instances_nll['predicted_cls_probs'][image_id]

        # Compute colors and predicted classes for energy score
        number_instances_en = predicted_box_means_en.shape[0]
        predicted_classes_en, assigned_colors_en, predicted_scores_em = compute_class_entropy(predicted_cls_probs_en,
                                                                                              cfg, class_list, 'en',
                                                                                              number_instances_en)
        # Compute colors and predicted classes for energy score
        number_instances_nll = predicted_box_means_nll.shape[0]
        predicted_classes_nll, assigned_colors_nll, predicted_scores_nll = compute_class_entropy(
            predicted_cls_probs_nll,
            cfg, class_list, 'nll',
            number_instances_nll)

        # Organize this better TODO
        # Exception to plot both instances with keys and without key
        try:
            predicted_instance_key_en = instances_en['predicted_instance_key'][image_id].cpu().numpy()
        except KeyError:
            predicted_instance_key_en = None
        try:
            predicted_instance_key_nll = instances_nll['predicted_instance_key'][image_id].cpu().numpy()
        except KeyError:
            predicted_instance_key_nll = None

        # Concat results in a single array
        predicted_box_means = concat_instances(predicted_box_means_en, predicted_box_means_nll)
        predicted_box_covariances = concat_instances(predicted_box_covariances_en, predicted_box_covariances_nll)
        assigned_colors = concat_instances(assigned_colors_en, assigned_colors_nll)
        predicted_classes = concat_instances(np.array(predicted_classes_en), np.array(predicted_classes_nll)).tolist()
        predicted_instance_key = concat_instances(predicted_instance_key_en, predicted_instance_key_nll)

        # Plot twice, one for EN and other for NLL
        plotted_detections = v.overlay_covariance_instances(boxes=predicted_box_means,
                                                            covariance_matrices=predicted_box_covariances,
                                                            assigned_colors=assigned_colors,
                                                            alpha=1.0,
                                                            labels=predicted_classes,
                                                            obj_key=predicted_instance_key)
        # cv2.imwrite('/home/sherlock/Desktop/ML/slam/data_process_slam/imgs_en_vs_nll/{}'.format(image_id), cv2.cvtColor(
        #         plotted_detections.get_image(),
        #         cv2.COLOR_RGB2BGR))
        cv2.imshow(
            'Detected Instances.',
            cv2.cvtColor(
                plotted_detections.get_image(),
                cv2.COLOR_RGB2BGR))
        cv2.waitKey()


def compute_class_entropy(predicted_cls_probs, cfg, class_list, loss_type, number_instances):
    if predicted_cls_probs.shape[0] > 0:
        if cfg.MODEL.META_ARCHITECTURE == "ProbabilisticGeneralizedRCNN" or cfg.MODEL.META_ARCHITECTURE == "ProbabilisticDetr":
            predicted_scores, predicted_classes = predicted_cls_probs[:, :-1].max(1)
        else:
            predicted_scores, predicted_classes = predicted_cls_probs.max(1)

        predicted_classes = predicted_classes.cpu().numpy()
        predicted_classes = [class_list[p_class] for p_class in predicted_classes]
        # RGBA, EN is Blue and nll is Yellow
        assigned_colors = np.array([0, 0, 1, 1]) if loss_type == 'en' else np.array([1, 1, 0, 1])
        assigned_colors = np.tile(assigned_colors, (number_instances, 1))
        predicted_scores = predicted_scores.cpu().numpy()
    else:
        predicted_scores = np.array([])
        predicted_classes = np.array([])
        assigned_colors = []
    return predicted_classes, assigned_colors, predicted_scores


def concat_instances(en_instances, nll_instalces):
    # TODO improve this
    instances = np.array([])
    if len(en_instances) == 0 or len(nll_instalces) == 0:
        if len(en_instances) == 0 and len(nll_instalces) == 0:
            return instances
        elif len(en_instances) == 0:
            instances = nll_instalces
        else:
            instances = en_instances
    else:
        instances = np.concatenate((en_instances, nll_instalces), axis=0)
    return instances


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
