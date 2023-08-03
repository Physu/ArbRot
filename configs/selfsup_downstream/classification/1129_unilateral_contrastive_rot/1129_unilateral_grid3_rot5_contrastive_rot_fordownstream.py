# 主要用于simplehead部分的代码测试
# 不包括复杂的头部结构，只用简单的网络实现对比学习和其他pretask 任务
# 但是包括了rgb和depth两个输入分支，用来分别完成对应的任务
norm_cfg = dict(type='BN', requires_grad=True)
# dataset_type = 'SUNRGBDDataset'
# data_root = 'data/sunrgbd/sunrgbd_trainval'
# img_norm_cfg = dict(
#     mean=[125.912, 116.549, 110.436],
#     std=[71.267, 73.216, 74.424],
#     to_rgb=True)
# label_norm_cfg = dict(
#     mean=[19050.278],
#     std=[9693.823],
# )
model = dict(
    type='Classification',
    # pretrained='/mnt/disk7/lhy/Cascade/mmsegmentation-0.14.1/newback/1022_r50_v2/iter_400000.pth',
    pretrained='/mnt/disk7/lhy/Cascade/mmsegmentation-0.14.1/newback/1123contrastive_rot/1123_unilateral_grid3_rot5_contrastive_rot/latest.pth',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        # contract_dilation=True,
        frozen_stages=4
    ),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=2048, num_classes=10),
    train_cfg=dict(),
    # neck=dict(
    #     type='NonLinearNeckV1',
    #     in_channels=2048,
    #     hid_channels=2048,
    #     out_channels=128,
    #     with_avg_pool=True),
    # head=dict(type='ContrastiveHead', temperature=0.2),
    # head_depth=dict(type='ContrastiveHead', temperature=0.2),
    # rot_head=dict(
    #     type='RotHead',
    #     in_channels=2048,
    #     num_classes=2,
    #     with_avg_pool=True
    # ),
    # loc_head=dict(
    #     type='LocHead',
    #     in_channels=2048,
    #     num_classes=9,
    #     with_avg_pool=True
    # ),
    # dep_head=dict(
    #     type='DepHead',
    #     in_channels=2048,
    #     out_channels=1,  # 输出dpeth image
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     out_depth_image_size=225
    # ),
    # img_head=dict(
    #     type='ImgHead',
    #     in_channels=2048,
    #     out_channels=3,  # 输出image
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     out_depth_image_size=225
    # ),
    test_cfg=dict(mode='whole'))


# dataset settings
dataset_type = 'Cifar10Dataset'
data_root = '/mnt/disk6/data/downloads/cifar10/'
img_norm_cfg = dict(mean=[125.307, 122.961, 113.8575],
                    std=[51.5865, 50.847, 51.255],
                    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='RandomCrop', crop_size=(32,32), padding=4),
    # dict(type='Resize', img_scale=(64, 32), ratio_range=(0.5, 2.0)),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img', 'gt_semantic_seg',
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Normalize',
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type='Cifar10Dataset',
        data_root='/mnt/disk6/data/downloads/cifar10',
        img_dir='train_images',
        ann_dir='train_labels',
        split='train_data_idx_backup.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            # dict(type='Resize', img_scale=(32, 64), ratio_range=(0.5, 2.0)),
            dict(type='RandomFlip', prob=0.5),
            # dict(type='RotateCircle_2imgs_2deps',
            #      resize=225,
            #      angle=4,
            #      grid=4,  # 注意 resize//grid 的结果表示分割的距离
            #      local_or_global='global',  # option:'local' or 'global'
            #      radius_ratio=1.0),
            dict(
                type='Normalize',
                mean=[125.307, 122.961, 113.8575],
                std=[51.5865, 50.847, 51.255],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_semantic_seg'
                ])
        ]),
    val=dict(
        type='Cifar10Dataset',
        data_root='/mnt/disk6/data/downloads/cifar10',
        img_dir='test_images',
        ann_dir='test_labels',
        split='test_data_idx_backup.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            # dict(type='LoadAnnotations'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(32, 32),
                # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                flip=False,
                transforms=[
                    # dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ])
        ]),
    test=dict(
        type='Cifar10Dataset',
        data_root='/mnt/disk6/data/downloads/cifar10',
        img_dir='test_images',
        ann_dir=' test_labels',
        split='test_data_idx_backup.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(32, 32),
                # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                flip=False,
                transforms=[
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ])
        ]))


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])


dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.00001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=120000)
checkpoint_config = dict(by_epoch=False, interval=40000)
evaluation = dict(interval=400000, metric='mIoU')
work_dir = 'newback/1129_unilateral_grid3_rot_contrastive_rot_downstream/1129_unilateral_grid3_rot5_contrastive_rot_fordownstream'
gpu_ids = range(0, 1)
