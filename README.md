# ArbRot

This is the 《Multi-modal Arbitrary Rotation based Self-supervised Learning for RGB-D Semantic Segmentation》 official implementation code.

For the ArbRot operation, please refer to ArbRot\mmseg\datasets\pipelines\transforms_img_hha.py RotateCircleImgAndHHA function

For the Multi-modal Multi-task Self-supervised Learning, please refer to: \ 
ArbRot\mmseg\models\segmentators\MoCoUnet0801.py for MoCo,  \
ArbRot\mmseg\models\segmentators\SimCLRUnet1026 for SimCLR, \
ArbRot\mmseg\models\segmentators\BYOLUnet1026 for BYOL, \
ArbRot\mmseg\models\segmentators\SimSiamUnet0623 for SimSiam 

## Usage
### Installation
1. Requirements

- Linux
- Python 3.6+
- PyTorch 1.7.0 or higher
- CUDA 10.0 or higher

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04.6 LTS
- CUDA: 10.0
- PyTorch 1.7.0
- Python 3.6.9

2. Install dependencies.
```shell
pip install -r requirements.txt
```
### Dataset
### Prepare SUN RGB-D Data

1. Download SUNRGBD v2 data [HERE](http://rgbd.cs.princeton.edu/data/) (SUNRGBD.zip, SUNRGBDMeta2DBB_v2.mat, SUNRGBDMeta3DBB_v2.mat) and the toolkits (SUNRGBDtoolbox.zip). Move all the downloaded files under OFFICIAL_SUNRGBD. Unzip the zip files.

2. Extract point clouds and annotations (class, v2 2D -- xmin,ymin,xmax,ymax, and 3D bounding boxes -- centroids, size, 2D heading) by running `extract_split.m`, `extract_rgbd_data_v2.m` and `extract_rgbd_data_v1.m` under the `matlab` folder.

We provide ready-made annotation files we generated offline for reference, which are provided by MMdetection3D. You can directly use these files for convenice.
|                                                        Dataset                                                         |                                                                                                           Train annotation file                                                                    |                                                                                                        Val annotation file                                                                             |
| :--------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                       SUN RGB-D                                                        |                                                               [sunrgbd_infos_train.pkl](https://download.openmmlab.com/mmdetection3d/data/sunrgbd/sunrgbd_infos_train.pkl)                         |                                             [sunrgbd_infos_val.pkl](https://download.openmmlab.com/mmdetection3d/data/sunrgbd/sunrgbd_infos_val.pkl)                                                   |      


Or download the converted dataset:
- [NYU-V2](https://drive.google.com/file/d/1VrRoWSxMkeJNSM12woiZEG_tN-N8ckj7/view?usp=sharing)


### Train
1. Config

    Edit config file in `./configs`.

   1. Set `data_root = path_to_dataset`. 
      
      E.g.,`data_root = "data/sunrgbd/sunrgbd_trainval"`.

2. Pretrain Run
    1. Distributed training
    ```shell
    ./tools/dist_train.sh config_path gpu_num --no-validate
    ```
    E.g., self-supervised train the model on SUN RGB-D with 2 GPUs, please run:
    ```shell
    .tools/dist_train_moco1.sh configs/selfsup/moco/from_scratch/1027_marc_res50_8b8_lr_1e-3_300e_scratch_byol_loc_rot_rgd_dgr.py 2 --no-validate
    ``` 
   2. Non-distributed training
    ```shell
    python tools/train.py config_path  --no-validate
    ```

## Finetuning on RGB-D segmentation:
We transfer the obtained weights of our Multi-modal Arbitrary Rotation based Self-supervised Learning to DeepLabv3 and ShapeConv(https://github.com/hanchaoleng/ShapeConv) methods as initialization.


### Acknowledgement

This repository is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation),[MMDetection3D](https://github.com/open-mmlab/mmdetection3d), [MMSelfsup](https://github.com/open-mmlab/mmselfsup), [ShapeConv](https://github.com/hanchaoleng/ShapeConv)
