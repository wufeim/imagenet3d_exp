# ImageNet3D Experiments

**ImageNet3D** dataset and helper code, from the following paper:

- [Overview](#overview)
- [Installation](#installation)
- [Preparing ImageNet3D Data](#preparing-imagenet3d-data)
- [Tasks](#tasks)
- [Citation](#citation)

## Overview

## Installation

```sh
conda create -n imagenet3d python=3.10
conda activate imagenet3d
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install pillow matplotlib opencv_python scipy tqdm omegaconf wandb
pip install -e .
```

## Preparing ImageNet3D Data

1. Download ImageNet3D data from [huggingface.co](https://huggingface.co/datasets/ccvl/ImageNet3D).

2. Preparing object-centric images.

    ```sh
    python3 scripts/preprocess_data.py --center_and_resize
    ```

## Tasks

### Linear Probing of Object-Level 3D Awareness

```sh
CUDA_VISIBLE_DEVICES=0 python3 scripts/linear_probing.py \
    --config configs/linear_probing_dinov2_vitb14.yaml \
    --exp_name linear_probing_dinov2_vitb14
```

Please refer to the `./configs` directory for configs for other backbones.

### Open-Vocabulary Pose Estimation

```sh
CUDA_VISIBLE_DEVICES=0 python3 scripts/open_pose_estimation.py \
    --config configs/open_pose_estimation_resnet50.yaml \
    --exp_name open_pose_estimation_resnet50
```

## Join Image Classification and Category-Level Pose Estimation

```sh
CUDA_VISIBLE_DEVICES=0 python3 scripts/pose_estimation_classification.py \
    --config configs/pose_estimation_classification_resnet50.yaml \
    --exp_name pose_estimation_classification_resnet50
```

## Citation

TBD.
