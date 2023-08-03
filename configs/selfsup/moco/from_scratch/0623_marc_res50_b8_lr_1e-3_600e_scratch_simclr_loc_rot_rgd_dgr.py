rotation_num = None
num_rot_classes = 12
grid_num = 25
# norm_cfg = dict(type='BN', requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
dataset_type = 'SUNRGBDHHADataset'
data_root = 'data/sunrgbd/sunrgbd_trainval'
work_dir = 'newback/moco/from_scratch/0622_marc_res50_b8_lr_1e-3_600e_from_scratch_moco_loc_rot_rgd_dgr_counterp'

# 0406 1.用了ssim 2.对于rot res 预测，smoothl1loss改为l1 loss
# 0411 1.将单纯的0-1.0，通过imagenet给归一化到ImageNet那个范围
# 0412 注意这里，重新修改了程序
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    to_rgb=True)
label_norm_cfg = dict(mean=[19050.278], std=[9693.823])
model = dict(
    type='MoCoUnet0412',
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[0, 1, 2, 3],
        norm_cfg=norm_cfg,
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')
    ),
    neck=dict(
        type='NonLinearNeck',  # 注意这里，为了适配simclr，进行了修改
        in_channels=2048,
        hid_channels=2048,
        out_channels=64,
        with_avg_pool=True),
    # ct_neck=dict(
    #     type='CTResNetNeck',
    #     in_channel=2048,
    #     num_deconv_filters=(256, 128, 64),
    #     num_deconv_kernels=(4, 4, 4),
    #     use_dcn=True),
    # ct_bbox_head=dict(
    #     type='CenterNetHead',
    #     num_classes=24,  # 原来80
    #     in_channel=64,
    #     feat_channel=64,
    #     loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
    #     loss_wh=dict(type='L1Loss', loss_weight=0.1),
    #     loss_offset=dict(type='L1Loss', loss_weight=1.0)),
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
    contrastive_head=dict(type='ContrastiveHead', temperature=0.1),
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
        # l1_loss=dict(type='L1Loss'),
        # l1_loss_weight=1.0,
        loss_total=True,
        # loss_total_weight=0.8
        ),
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
        # l1_loss=dict(type='L1Loss'),
        # l1_loss_weight=1.0,
        loss_total=True,
        # loss_total_weight=0.2,
        ),
    # imitative_pretrained='/home/lhy/.cache/torch/hub/checkpoints/resnet101-5d3b4d8f.pth',
    # imitative_loss=1
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
            dict(type='NormalizeImgAndHHA', img_norm_cfg=img_norm_cfg),
            # dict(type='Normalize', **img_norm_cfg),
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

# optimizer
optimizer = dict(type='SGD', lr=0.001, weight_decay=1e-4, momentum=0.9)
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb

# learning policy
# lr_config = dict(policy='CosineAnnealing', min_lr=0.)
# optimizer = dict(type='SGD', lr=0.002, weight_decay=0.0001, momentum=0.9)
# optimizer_config = dict()
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.0001,
    step=[19379*2*3*2, 29068*2*3*2])
runner = dict(type='IterBasedRunner', max_iters=32297*2*3*2)
checkpoint_config = dict(interval=30000)

log_config = dict(
    interval=120,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# optimizer_config = dict()
# # learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
dist_params = dict(backend='nccl')
cudnn_benchmark = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 1)

