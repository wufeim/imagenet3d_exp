import argparse
import json
import os

import numpy as np
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='open_vocab_pose_resnet50')
    parser.add_argument('--output_dir', type=str, default='exp')
    parser.add_argument('--output_filename', type=str, default='evaluate_epo120.json')
    parser.add_argument('--meta_class_file', type=str, default='/home/wufeim/research/imagenet3d_data/categories/imagenet3d_metaclasses_189.json')
    parser.add_argument('--known_classes', type=str, default='/home/wufeim/research/imagenet3d_data/categories/knowns.txt')
    parser.add_argument('--storage_file', type=str, default='open_vocab_imagenet3d_0530_189.json')
    return parser.parse_args()


def main(args):
    assert os.path.isdir(args.output_dir)

    meta_classes = json.load(open(args.meta_class_file))
    known_classes = open(args.known_classes).read().strip().split('\n')

    output_json = json.load(open(os.path.join(args.output_dir, args.output_filename)))

    sample_names = output_json['sample_names']
    categories = output_json['categories']
    pose_errors = output_json['pose_errors']

    performance = {'known_avg': {}}
    best_pose_errors = np.array([e for e, c in zip(pose_errors, categories) if c in known_classes])
    _categories = [c for c in categories if c in known_classes]
    performance['known_avg']['pi_6_acc'] = np.mean(best_pose_errors < np.pi/6) * 100
    performance['known_avg']['pi_18_acc'] = np.mean(best_pose_errors < np.pi/18) * 100
    performance['known_avg']['mean_err'] = np.mean(best_pose_errors) / np.pi * 180.0
    performance['known_avg']['median_err'] = np.median(best_pose_errors) / np.pi * 180.0
    for k in meta_classes:
        _pose_errors = np.array([err for c, err in zip(_categories, best_pose_errors) if c in meta_classes[k]])
        assert len(_pose_errors) > 0, 'known_'+k
        performance['known_'+k] = {}
        performance['known_'+k]['pi_6_acc'] = np.mean(_pose_errors < np.pi/6) * 100
        performance['known_'+k]['pi_18_acc'] = np.mean(_pose_errors < np.pi/18) * 100
        performance['known_'+k]['mean_err'] = np.mean(_pose_errors) / np.pi * 180.0
        performance['known_'+k]['meadian_err'] = np.median(_pose_errors) / np.pi * 180.0

    performance['unknown_avg'] = {}
    best_pose_errors = np.array([e for e, c in zip(pose_errors, categories) if c not in known_classes])
    _categories = [c for c in categories if c in known_classes]
    performance['unknown_avg']['pi_6_acc'] = np.mean(best_pose_errors < np.pi/6) * 100
    performance['unknown_avg']['pi_18_acc'] = np.mean(best_pose_errors < np.pi/18) * 100
    performance['unknown_avg']['mean_err'] = np.mean(best_pose_errors) / np.pi * 180.0
    performance['unknown_avg']['median_err'] = np.median(best_pose_errors) / np.pi * 180.0
    for k in meta_classes:
        _pose_errors = np.array([err for c, err in zip(_categories, best_pose_errors) if c in meta_classes[k]])
        assert len(_pose_errors) > 0, 'unknown_'+k
        performance['unknown_'+k] = {}
        performance['unknown_'+k]['pi_6_acc'] = np.mean(_pose_errors < np.pi/6) * 100
        performance['unknown_'+k]['pi_18_acc'] = np.mean(_pose_errors < np.pi/18) * 100
        performance['unknown_'+k]['mean_err'] = np.mean(_pose_errors) / np.pi * 180.0
        performance['unknown_'+k]['meadian_err'] = np.median(_pose_errors) / np.pi * 180.0

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
