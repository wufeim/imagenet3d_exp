import argparse
import json
import os

import numpy as np
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='pose_estimation_classification_swintranst')
    parser.add_argument('--output_dir', type=str, default='exp')
    parser.add_argument('--output_filename', type=str, default='evaluate_epo120.json')
    parser.add_argument('--meta_class_file', type=str, default='/home/wufeim/research/imagenet3d_data/categories/imagenet3d_metaclasses_189.json')
    parser.add_argument('--storage_file', type=str, default='pose_cls_imagenet3d_0530_189.json')
    return parser.parse_args()


def main(args):
    assert os.path.isdir(args.output_dir)

    meta_classes = json.load(open(args.meta_class_file))

    output_json = json.load(open(os.path.join(args.output_dir, args.output_filename)))

    sample_names = output_json['sample_names']
    categories = output_json['categories']
    pose_errors = output_json['pose_errors']
    classifications = output_json['classifications']

    performance = {'avg': {}}

    best_pose_errors = np.array(pose_errors) + (1 - np.array(classifications)) * np.pi
    performance['avg']['pi_6_acc'] = np.mean(best_pose_errors < np.pi/6) * 100
    performance['avg']['pi_18_acc'] = np.mean(best_pose_errors < np.pi/18) * 100
    performance['avg']['mean_err'] = np.mean(best_pose_errors) / np.pi * 180.0
    performance['avg']['median_err'] = np.median(best_pose_errors) / np.pi * 180.0
    for k in meta_classes:
        _pose_errors = np.array([err for c, err in zip(categories, best_pose_errors) if c in meta_classes[k]])
        performance[k] = {}
        performance[k]['pi_6_acc'] = np.mean(_pose_errors < np.pi/6) * 100
        performance[k]['pi_18_acc'] = np.mean(_pose_errors < np.pi/18) * 100
        performance[k]['mean_err'] = np.mean(_pose_errors) / np.pi * 180.0
        performance[k]['meadian_err'] = np.median(_pose_errors) / np.pi * 180.0

    if os.path.isfile(args.storage_file):
        storage = json.load(open(os.path.join(args.storage_file)))
    else:
        storage = {}
    storage[args.exp_name] = performance
    with open(args.storage_file, 'w') as fp:
        json.dump(storage, fp, indent=4)


if __name__ == '__main__':
    args = parse_args()
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    main(args)
