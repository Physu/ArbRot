rotation_num = None
num_rot_classes = 12
grid_num = 9
norm_cfg = dict(type='BN', requires_grad=True)
dataset_type = 'SUNRGBDHHADataset'
data_root = 'data/sunrgbd/sunrgbd_trainval'
work_dir = 'newback/moco/0405_marcl_decouple_resnext101_32_8d_bsize_8_lr_5e-4'

label_norm_cfg = dict(mean=[19050.278], std=[9693.823])
model = dict(
    type='MoCoUnet0405',
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=8,
        in_channels=3,
        out_indices=[0, 1, 2, 3],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='NonLinearNeckV1',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    loc_head=dict(
        type='LocHead0405',
        in_channels=2048,
        num_classes=grid_num,
        with_avg_pool=True),
    rot_head=dict(
        type='RotHead0403',
        in_channels=2048,
        num_rot_bins=24,
        with_avg_pool=True,
        rot_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        rot_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        shared_conv_channels=(512, 128),
        num_rot_out_channels=24,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.1),
        bias=True),
    contrastive_head=dict(type='ContrastiveHead', temperature=0.2),
    generation_neck=dict(
        type='GenerationNeck',
        unethead_in_channels=3,
        base_channels=64,
        num_stages=4,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    rgb_generate_depth_head=dict(
        type='RGBGenerateDepthHead',
        with_avg_pool=False,
        in_channels=256,
        conv_channels=(128, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        # ssim_loss=dict(type='DirectionAwareSSIM_Loss'),
        # ssim_weight=1.0,
        # edge_loss=dict(type='EdgeAwareLoss'),
        # edge_weight=0.7,
        l1_loss=dict(type='L1Loss'),
        l1_loss_weight=1.0),
    depth_generate_rgb_head=dict(
        type='DepthGenerateRGBHead',
        with_avg_pool=False,
        in_channels=256,
        conv_channels=(128, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        # ssim_loss=dict(type='DirectionAwareSSIM_Loss'),
        # ssim_weight=1.0,
        # edge_loss=dict(type='EdgeAwareLoss'),
        # edge_weight=0.7,
        l1_loss=dict(type='L1Loss'),
        l1_loss_weight=1.0)
)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='SUNRGBDHHADataset',
        data_root='data/sunrgbd/sunrgbd_trainval',
        img_dir='image',
        ann_dir='hha_bfx',
        split='trainval_data_idx_backup.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadSUNRGBDAnnotations', keep_origin=True),
            dict(
                type='RotateCircleImgAndHHA',
                resize=256,
                angle=rotation_num,
                grid=grid_num,
                local_or_global='global',
                save_augmented=False,
                log_dir=work_dir),
            dict(type='RandomFlipImgHHA', prob=0.5),
            dict(type='NormalizeImgAndHHA'),
            dict(type='DefaultFormatBundle_SUNRGBDHHA'),
            dict(
                type='Collect',
                keys=[
                    'img_aug1', 'img_aug2', 'gt_semantic_seg_aug1',
                    'gt_semantic_seg_aug2', 'sunrgbd_rotation1',
                    'sunrgbd_rotation2', 'paste_location1', 'paste_location2'
                ])
        ]),
    val=dict(
        type='SUNRGBDHHADataset',
        data_root='data/sunrgbd/sunrgbd_trainval',
        img_dir='image',
        ann_dir='hha_bfx',
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
        type='SUNRGBDHHADataset',
        data_root='data/sunrgbd/sunrgbd_trainval',
        img_dir='image',
        ann_dir='hha_bfx',
        split='val_data_idx_backup.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Normalize_img_and_label',
                img_norm_cfg=dict(
                    mean=[125.912, 116.549, 110.436],
                    std=[71.267, 73.216, 74.424],
                    to_rgb=True),
                label_norm_cfg=dict(mean=[19050.278], std=[9693.823])),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
optimizer = dict(type='SGD', lr=0.0005, weight_decay=0.0001, momentum=0.9)
optimizer_config = dict()
runner = dict(type='IterBasedRunner', max_iters=32297*2)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.0001,
    step=[19379*2, 29068*2])
checkpoint_config = dict(interval=20000)
log_config = dict(
    interval=30,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
cudnn_benchmark = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 1)

