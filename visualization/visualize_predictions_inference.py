import cv2
import numpy as np
import os
import ujson as json
import core
import sys

from scipy.stats import entropy
from matplotlib import cm

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
def main(
        args,
        cfg=None):
    # Setup config
    if cfg is None:
        cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)

    cfg.defrost()
    cfg.ACTUAL_TEST_DATASET = args.test_dataset

    # Update value min_allowed_score and inference_output_dir
    inference_output_dir = args.output_dir
    min_allowed_score = args.min_allowed_score

    # get preprocessed instances
    preprocessed_predicted_instances, preprocessed_gt_instances = evaluation_utils_inference.get_per_frame_preprocessed_instances(
        cfg, inference_output_dir, min_allowed_score)

    # get metacatalog and image infos
    meta_catalog = MetadataCatalog.get(args.test_dataset)
    print(meta_catalog)
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

        predicted_box_means = preprocessed_predicted_instances['predicted_boxes'][image_id].cpu(
        ).numpy()
        predicted_box_covariances = preprocessed_predicted_instances[
            'predicted_covar_mats'][image_id].cpu(
        ).numpy()

        predicted_cls_probs = preprocessed_predicted_instances['predicted_cls_probs'][image_id]

        if predicted_cls_probs.shape[0] > 0:
            if cfg.MODEL.META_ARCHITECTURE == "ProbabilisticGeneralizedRCNN" or cfg.MODEL.META_ARCHITECTURE == "ProbabilisticDetr":
                predicted_scores, predicted_classes = predicted_cls_probs[:, :-1].max(
                    1)
                predicted_entropies = entropy(
                    predicted_cls_probs.cpu().numpy(), base=2)

            else:
                predicted_scores, predicted_classes = predicted_cls_probs.max(
                    1)
                predicted_entropies = entropy(
                    np.stack(
                        (predicted_scores.cpu().numpy(),
                         1 - predicted_scores.cpu().numpy())),
                    base=2)
            predicted_classes = predicted_classes.cpu(
            ).numpy()
            predicted_classes = [class_list[p_class]
                                 for p_class in predicted_classes]
            assigned_colors = cm.autumn(predicted_entropies)
            predicted_scores = predicted_scores.cpu().numpy()
        else:
            predicted_scores = np.array([])
            predicted_classes = np.array([])
            assigned_colors = []

        # Exception to plot both instances with keys and without key
        try:
            predicted_instance_key = preprocessed_predicted_instances['predicted_instance_key'][image_id].cpu().numpy()
        except KeyError:
            predicted_instance_key = None

        plotted_detections = v.overlay_covariance_instances(
            boxes=predicted_box_means,
            covariance_matrices=predicted_box_covariances,
            assigned_colors=assigned_colors,
            alpha=1.0,
            labels=predicted_classes,
            obj_key=predicted_instance_key)
        # cv2.imwrite('/home/sherlock/Desktop/ML/slam/data_process_slam/detr_nll/{}'.format(image_id), cv2.cvtColor(
        #         plotted_detections.get_image(),
        #         cv2.COLOR_RGB2BGR))
        # cv2.imshow(
        #     'Detected Instances.',
        #     cv2.cvtColor(
        #         plotted_detections.get_image(),
        #         cv2.COLOR_RGB2BGR))
        # cv2.waitKey()


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
