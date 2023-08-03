norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoderSUNRGBD',
    pretrained='open-mmlab://resnet50_v1c',
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
    decode_head=dict(
        type='ASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='MSELoss')),
    auxiliary_head=dict(
        type='FCNHeadSUNRGBD',
        in_channels=(1024, 2048),
        input_transform='multiple_select',
        in_index=(2, 3),
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='MSELoss'),
        loss_rotation=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
        loss_location=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'SUNRGBDDataset'
data_root = 'data/sunrgbd/sunrgbd_trainval'
img_norm_cfg = dict(
    mean=[125.912, 116.549, 110.436],
    std=[71.267, 73.216, 74.424],
    to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSUNRGBDAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='ResizeRotatePaste', resize=(225, 225), small_resize=(75, 75)),
    dict(
        type='Normalize',
        mean=[125.912, 116.549, 110.436],
        std=[71.267, 73.216, 74.424],
        to_rgb=True),
    dict(type='DefaultFormatBundle_SUNRGBD'),
    dict(
        type='Collect',
        keys=[
            'img', 'gt_semantic_seg', 'sunrgbd_rotation', 'sunrgbd_location'
        ])
]
test_pipeline = [
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
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='SUNRGBDDataset',
        data_root='data/sunrgbd/sunrgbd_trainval',
        img_dir='image',
        ann_dir='depth_bfx_png',
        split='train_data_idx_backup.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadSUNRGBDAnnotations'),
            dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='ResizeRotatePaste',
                resize=(225, 225),
                small_resize=(75, 75)),
            dict(
                type='Normalize',
                mean=[125.912, 116.549, 110.436],
                std=[71.267, 73.216, 74.424],
                to_rgb=True),
            dict(type='DefaultFormatBundle_SUNRGBD'),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_semantic_seg', 'sunrgbd_rotation',
                    'sunrgbd_location'
                ])
        ]),
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
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=120000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=120001, metric='mIoU')
work_dir = 'newback/0825_deeplabv3_r50-d8_512x512_20k_sunrgbd/'
gpu_ids = range(0, 1)
