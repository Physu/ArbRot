angle_div = None  # None表示 360无限制旋转，180 表示2°一份【0，2，4……，358】
num_rot_classes = 12
grid_num = 3  # 没啥用了，后面可以删掉了
norm_cfg = dict(type='BN', requires_grad=True)
# norm_cfg = dict(type='SyncBN', requires_grad=True)
dataset_type = 'SUNRGBDHHADataset'
data_root = 'data/sunrgbd/sunrgbd_trainval'
work_dir = 'newback/ArbRotPlus/0512_ArbRotPlus_res50_2b4_lr_1e-1_100e_type2_aug12_dwa_cosinelr'

# 0406 1.用了ssim 2.对于rot res 预测，smoothl1loss改为l1 loss
# 0411 1.将单纯的0-1.0，通过imagenet给归一化到ImageNet那个范围
# 0412 注意这里，重新修改了程序
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    to_rgb=True)
model = dict(
    type='MoCoUnetDWAPlus',
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
        type='NonLinearNeckV1',
        in_channels=2048,
        hid_channels=2048,
        out_channels=64,
        with_avg_pool=True),
    loc_neck=dict(
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
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1, inplace=False),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    loc_bbox_head=dict(
        type='LocHeadPlus',
        num_classes=1,
        in_channel=256,
        feat_channel=64,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=0.1),
        loss_wh=dict(type='L1Loss', loss_weight=0.01),
        loss_offset=dict(type='L1Loss', loss_weight=0.1),
        output_flag=False),  # output_flag 是否输出预测的boxes的位置信息
    rot_head=dict(
        type='RotHeadPlus',
        in_channels=256,
        num_rot_bins=24,
        with_avg_pool=True,
        rot_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        rot_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        conv_cfg=dict(type='Conv1d'),
        loss_total_weight=1.0*0.8),
    contrastive_head=dict(type='ContrastiveHead', temperature=0.2, loss_total_weight=0.5*0.2),
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
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1, inplace=False),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    rgb_generate_depth_head=dict(
        type='RGBGenerateDepthHead',
        with_avg_pool=False,
        in_channels=256,
        conv_channels=(128, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1, inplace=False),
        # ssim_loss=dict(type='DirectionAwareSSIM_Loss'),
        # ssim_weight=1.0,
        # edge_loss=dict(type='EdgeAwareLoss'),
        # edge_weight=0.7,
        # l1_loss=dict(type='L1Loss'),
        # l1_loss_weight=1.0,
        loss_total=True,
        loss_total_weight=0.5*0.8
        ),
    depth_generate_rgb_head=dict(
        type='DepthGenerateRGBHead',
        with_avg_pool=False,
        in_channels=256,
        conv_channels=(128, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1, inplace=False),
        # ssim_loss=dict(type='DirectionAwareSSIM_Loss'),
        # ssim_weight=1.0,
        # edge_loss=dict(type='EdgeAwareLoss'),
        # edge_weight=0.7,
        # l1_loss=dict(type='L1Loss'),
        # l1_loss_weight=1.0,
        loss_total=True,
        loss_total_weight=0.5*0.2,
        ),
)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='SUNRGBDHHADataset',
        data_root='data/sunrgbd/sunrgbd_trainval',
        img_dir='image',
        ann_dir='hha_bfx',
        split='train_data_idx_backup.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadSUNRGBDAnnotations', keep_origin=True),
            dict(type='RandomCropImgHHA',  #1
                 crop_size=(500, 500),
                 padding=0),
            # dict(type='ColorJitterImgHHA',  #2
            #      brightness=0.4,
            #      contrast=0.4,
            #      saturation=0.4),
            # dict(type='RandomGrayscaleImgHHA',  #3
            #      gray_prob=0.1),
            # dict(type='GaussianBlurImgHHA', sigma_min=0.1, sigma_max=2.0, p=0.5),  #4
            # dict(type='RandomFlipImgHHABefore', prob=0.5),  #5
            dict(
                type='RotateCircleImgAndHHAArbRotPlus',
                resize=256,
                angle=angle_div,
                grid=grid_num,
                local_or_global='global',
                save_augmented=False,
                log_dir=work_dir),
            # dict(type='RandomFlipImgHHA', prob=0.5),
            dict(type='NormalizeImgAndHHA', img_norm_cfg=img_norm_cfg),
            # dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle_SUNRGBDHHA'),
            dict(
                type='Collect',
                keys=[
                    'img_aug1', 'img_aug2', 'gt_semantic_seg_aug1',
                    'gt_semantic_seg_aug2', 'sunrgbd_rotation1',
                    'sunrgbd_rotation2', 'paste_location1', 'paste_location2',
                    'ori_img', 'ori_gt_semantic_seg',
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
optimizer = dict(type='SGD', constructor='DefaultOptimizerConstructorRevise', lr=0.1, weight_decay=1e-4, momentum=0.9)
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb

# learning policy
# lr_config = dict(policy='CosineAnnealing', min_lr=0.)
# optimizer = dict(type='SGD', lr=0.002, weight_decay=0.0001, momentum=0.9)
# optimizer_config = dict()
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr=0)
# runner = dict(type='IterBasedRunner', max_iters=32297*2)
runner = dict(type='IterBasedRunner', max_iters=33032*2)  # 两卡，bs=8，则100epoch iter为33032
checkpoint_config = dict(interval=33032)

log_config = dict(
    interval=30,
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

