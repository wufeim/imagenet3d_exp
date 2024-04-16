# ImageNet3D

**ImageNet3D** dataset and helper code, from the following paper:

## Overview

## Installation

```sh
conda create -n imagenet3d python=3.10
conda activate imagenet3d
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install pillow matplotlib opencv_python scipy tqdm omegaconf wandb
pip install -e .
```

## Download ImageNet3D

Modify the `local_dir` parameter to your local directory.

```py
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='ccvl/imagenet3d-0409',
    repo_type='dataset',
    filename='imagenet3d_0409.zip',
    local_dir='/path/to/imagenet3d_0409.zip',
    local_dir_use_symlinks=False)
```

## Citation
