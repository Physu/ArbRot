# 主要用于simplehead部分的代码测试
# 不包括复杂的头部结构，只用简单的网络实现对比学习和其他pretask 任务
# 但是包括了rgb和depth两个输入分支，用来分别完成对应的任务
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
    type='SUNRGBDMOCOSimpleHeadUnilateralConcate',
    pretrained='torchvision://resnet50',
    queue_len=4096,
    feat_dim=128,
    momentum=0.999,
    dataset_type='SUNRGBDMOCODataset',
    data_root='data/sunrgbd/sunrgbd_trainval',

    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    # neck=dict(
    #     type='NonLinearNeckV3',
    #     in_channels=2048,
    #     hid_channels=2048,
    #     out_channels=128,
    #     with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.2, contrastive_weight=0.1),
    head_depth=dict(type='ContrastiveHead', temperature=0.2, contrastive_weight=0.1),
    rot_head=dict(
        type='RotHead',
        in_channels=2048,
        num_classes=12,
        with_avg_pool=True,
        rot_weight=0.2  # 用来控制loss 权重
    ),
    loc_head=dict(
        type='LocHead',
        in_channels=2048,
        num_classes=9,
        with_avg_pool=True,
        loc_weight=0.2
    ),
    # ssim_loss=dict(type='DirectionAwareSSIM_Loss'),
    # edge_aware_loss=dict(type='EdgeAwareLoss'),
    rgb_generate_dep_head=dict(
        type='RGBGenerateDepHead',
        in_channels=2048,
        out_channels=1,  # 输出dpeth image
        norm_cfg=dict(type='BN', requires_grad=True),
        out_depth_image_size=256,
        # ssim_loss=dict(type='DirectionAwareSSIM_Loss'),
        edge_loss=dict(type='EdgeAwareLoss'),
        edge_weight=0.7,  # l1loss * edge_weight + edgeloss * (1-edgeweight)
        label_norm_cfg=label_norm_cfg,
    ),
    dep_generate_rgb_head=dict(
        type='DepGenerateRGBHead',
        in_channels=2048,
        out_channels=3,  # 输出image
        norm_cfg=dict(type='BN', requires_grad=True),
        out_rgb_image_size=256,
        ssim_loss=dict(type='DirectionAwareSSIM_Loss'),
        ssim_weight=0.84,
        # edge_loss=dict(type='EdgeAwareLoss')
        img_norm_cfg=img_norm_cfg
    ),
    test_cfg=dict(mode='whole'))


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,  # 关于这个worker设置：4*gpu_num https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/3
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
            # 不要角度的影响，只考虑位置的影响
            dict(type='RotateCircle',
                 resize=256,
                 angle=12,
                 grid=3,  # 注意 resize//grid 的结果表示分割的距离
                 local_or_global='global',  # option:'local' or 'global'
                 save_augmented=False),
            ##########################################################
            # dict(  # 这个可是有点问题，需要调整，颜色调整的太厉害了
            #     type='ColorJitter',
            #     brightness=0.4,
            #     contrast=0.4,
            #     saturation=0.4,
            #     prepare_for_moco=True),
            # dict(type='RandomGrayscale', p=0.2),  # 表示有0.2的概率，变为灰度图
            # dict(
            #     type='GaussianBlur',
            #     sigma_min=0.1,
            #     sigma_max=2.0,
            #     prepare_for_moco=True),
            #########################################################
            dict(
                type='Normalize_img_and_label',
                img_norm_cfg=img_norm_cfg,
                label_norm_cfg=label_norm_cfg),
            dict(type='DefaultFormatBundle_SUNRGBD'),
            dict(
                type='Collect',  # 虽然关键词里没有img，但是最后输入到backbone里面还是有img这个关键词，具体实现在mmseg/datasets/sunrgbd_moco.py 中的 prepare_train_img 方法
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
optimizer = dict(type='SGD', lr=0.00001, weight_decay=0.0001, momentum=0.9)
optimizer_config = dict()
runner = dict(type='IterBasedRunner', max_iters=66063)  # 200 epoch
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)
checkpoint_config = dict(interval=30000)
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
work_dir = 'newback/unilateral_moco/1212_unilateral_new_design/1212_unilateral_grid2_only_contrastive_lre-5_ssim_edge_loss'