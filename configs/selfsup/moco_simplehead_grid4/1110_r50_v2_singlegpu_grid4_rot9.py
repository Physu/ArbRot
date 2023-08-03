# 主要用于simplehead部分的代码测试
# 不包括复杂的头部结构，只用简单的网络实现对比学习和其他pretask 任务
norm_cfg = dict(type='BN', requires_grad=True)
dataset_type = 'SUNRGBDDataset'
data_root = 'data/sunrgbd/sunrgbd_trainval'
img_norm_cfg = dict(
    mean=[125.912, 116.549, 110.436],
    std=[71.267, 73.216, 74.424],
    to_rgb=True)
label_norm_cfg = dict(
    mean=[19050.278],
    std=[9693.823],
)
model = dict(
    type='SUNRGBDMOCOSimpleHead',
    # type='EncoderDecoderSUNRGBD',
    pretrained='open-mmlab://resnet50_v1c',
    queue_len=4096,
    feat_dim=128,
    momentum=0.999,
    dataset_type='SUNRGBDMOCODataset',
    data_root='data/sunrgbd/sunrgbd_trainval',
    label_norm_cfg=label_norm_cfg,
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
        contract_dilation=True),
    # decode_head=dict(
    #     type='ASPPHeadSUNRGBD',
    #     in_channels=2048,
    #     in_index=3,
    #     channels=512,
    #     dilations=(1, 12, 24, 36),
    #     dropout_ratio=0.1,
    #     num_classes=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     align_corners=False,
    #     loss_decode=dict(type='MSELoss')),
    # auxiliary_head=dict(
    #     type='FCNHeadSUNRGBD',
    #     in_channels=(1024, 2048),
    #     input_transform='multiple_select',
    #     in_index=(2, 3),  # 这个表示在backbone出来的四个层次的特征中选取第3，4个特征
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     align_corners=False,
    #     # loss_decode=dict(type='MSELoss'),
    #     loss_rotation=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
    #     loss_location=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
    #     num_angle=8,
    #     num_location=9
    # ),
    neck=dict(
        type='NonLinearNeckV1',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.2),
    rot_head=dict(
        type='RotHead',
        in_channels=2048,
        num_classes=9,
        with_avg_pool=True
    ),
    loc_head=dict(
        type='LocHead',
        in_channels=2048,
        num_classes=16,
        with_avg_pool=True
    ),
    dep_head=dict(
        type='DepHead',
        in_channels=2048,
        out_channels=1,  # 输出dpeth image
        norm_cfg=dict(type='BN', requires_grad=True),
        out_depth_image_size=225
    ),
    test_cfg=dict(mode='whole'))

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type='SUNRGBDMOCODataset',
        data_root='data/sunrgbd/sunrgbd_trainval',
        img_dir='image',
        ann_dir='depth_bfx_png',
        split='train_data_idx_backup.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadSUNRGBDAnnotations', keep_origin=True),  # keep_origin 是否保持深度信息，还是uint8形式
            # dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomFlip', prob=0.5),
            dict(type='RotateCircle',
                 resize=225,
                 angle=9,
                 grid=4,  # 注意 resize//grid 的结果表示分割的距离
                 local_or_global='global',  # option:'local' or 'global'
                 radius=37),
            ##########################################################
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                prepare_for_moco=True),
            # dict(type='RandomGrayscale', p=0.2),  # 表示有0.2的概率，变为灰度图
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0,
                prepare_for_moco=True),
            #########################################################
            dict(
                type='Normalize_img_and_label',
                img_norm_cfg=img_norm_cfg,
                label_norm_cfg=label_norm_cfg),
            dict(type='DefaultFormatBundle_SUNRGBD'),
            dict(
                type='Collect',
                keys=[
                    'img_aug1', 'img_aug2',
                    'gt_semantic_seg_aug1', 'gt_semantic_seg_aug2',
                    'sunrgbd_rotation1', 'sunrgbd_rotation2',
                    'paste_location1', 'paste_location2'
                ])
        ],
    ),
    val=dict(
        type='SUNRGBDDataset',
        data_root='data/sunrgbd/sunrgbd_trainval',
        img_dir='image',
        ann_dir='depth_bfx_png',
        split='val_data_idx_backup.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[125.912, 116.549, 110.436],
                        std=[71.267, 73.216, 74.424],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='SUNRGBDDataset',
        data_root='data/sunrgbd/sunrgbd_trainval',
        img_dir='image',
        ann_dir='depth_bfx_png',
        split='val_data_idx_backup.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[125.912, 116.549, 110.436],
                        std=[71.267, 73.216, 74.424],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
# optimizer
optimizer = dict(type='SGD', lr=0.003, weight_decay=0.0001, momentum=0.9)
optimizer_config = dict()
runner = dict(type='IterBasedRunner', max_iters=6607)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)
checkpoint_config = dict(interval=3000)
# runtime settings

# yapf:disable
log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# yapf:enable
# runtime settings
dist_params = dict(backend='nccl')
cudnn_benchmark = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 1)
