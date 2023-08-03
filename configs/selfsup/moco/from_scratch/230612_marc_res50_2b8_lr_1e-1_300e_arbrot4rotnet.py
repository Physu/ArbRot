rotation_num = 4
num_rot_classes = 12
grid_num = 25
# norm_cfg = dict(type='BN', requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
dataset_type = 'SUNRGBDHHADataset'
data_root = 'data/sunrgbd/sunrgbd_trainval'
work_dir = 'newback/moco/from_scratch/20230612_marc_res50_2b8_lr_1e-1_300e_arbrot4_rotnet'
# 1110 之前的版本，太傻叉了自己，搞错了学习率这个问题，娘希匹
# 0406 1.用了ssim 2.对于rot res 预测，smoothl1loss改为l1 loss
# 0411 1.将单纯的0-1.0，通过imagenet给归一化到ImageNet那个范围
# 0412 注意这里，重新修改了程序
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    to_rgb=True)
label_norm_cfg = dict(mean=[19050.278], std=[9693.823])
model = dict(
    type='MoCoUnet0903',
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
    # neck=dict(
    #     type='NonLinearNeckSimSiam',  # 注意这里，为了适配simclr，进行了修改
    #     in_channels=2048,
    #     hid_channels=2048,
    #     out_channels=2048,
    #     num_layers=3,
    #     with_last_bn_affine=False,
    #     with_avg_pool=True),
    # loc_head=dict(
    #     type='LocHead0405',
    #     in_channels=2048,
    #     num_classes=grid_num,
    #     with_avg_pool=True,
    #     loss_total_weight=0.1),
    rot_head=dict(
        type='RotHead4Pred',
        in_channels=2048,
        num_rot_bins=4,
        with_avg_pool=True,
        rot_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        rot_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        shared_conv_channels=(512, 128),
        num_rot_out_channels=4,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.1),
        bias=True,
        loss_total_weight=1.),
    # contrastive_head=dict(  # 适用于BYOL
    #     type='LatentPredictHead',
    #     predictor=dict(
    #         type='NonLinearNeck',
    #         in_channels=2048,
    #         hid_channels=512,
    #         out_channels=2048,
    #         with_avg_pool=False,
    #         with_last_bn=False,
    #         with_last_bias=True),
    #     loss_total_weight=0.2),
    # generation_neck=dict(
    #     type='GenerationNeck',
    #     unethead_in_channels=3,
    #     base_channels=64,
    #     num_stages=4,
    #     strides=(1, 1, 1, 1, 1),
    #     enc_num_convs=(2, 2, 2, 2, 2),
    #     dec_num_convs=(2, 2, 2, 2),
    #     downsamples=(True, True, True, True),
    #     enc_dilations=(1, 1, 1, 1, 1),
    #     dec_dilations=(1, 1, 1, 1),
    #     with_cp=False,
    #     conv_cfg=None,
    #     act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
    #     upsample_cfg=dict(type='InterpConv'),
    #     norm_eval=False),
    # rgb_generate_depth_head=dict(
    #     type='RGBGenerateDepthHead',
    #     with_avg_pool=False,
    #     in_channels=256,
    #     conv_channels=(128, 3),
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
    #     # ssim_loss=dict(type='DirectionAwareSSIM_Loss'),
    #     # ssim_weight=1.0,
    #     # edge_loss=dict(type='EdgeAwareLoss'),
    #     # edge_weight=0.7,
    #     # l1_loss=dict(type='L1Loss'),
    #     # l1_loss_weight=1.0,
    #     loss_total=True,
    #     loss_total_weight=0.45
    # ),
    # depth_generate_rgb_head=dict(
    #     type='DepthGenerateRGBHead',
    #     with_avg_pool=False,
    #     in_channels=256,
    #     conv_channels=(128, 3),
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
    #     # ssim_loss=dict(type='DirectionAwareSSIM_Loss'),
    #     # ssim_weight=1.0,
    #     # edge_loss=dict(type='EdgeAwareLoss'),
    #     # edge_weight=0.7,
    #     # l1_loss=dict(type='L1Loss'),
    #     # l1_loss_weight=1.0,
    #     loss_total=True,
    #     loss_total_weight=0.2,
    # ),
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
            # dict(type='RandomCropImgHHA',  #1
            #      crop_size=(500, 500),
            #      padding=0),
            # dict(type='ColorJitterImgHHA',  #2
            #      brightness=0.4,
            #      contrast=0.4,
            #      saturation=0.4),
            # dict(type='RandomGrayscaleImgHHA',  #3
            #      gray_prob=0.1),
            # dict(type='GaussianBlurImgHHA', sigma_min=0.1, sigma_max=2.0, p=0.5),  #4
            # dict(type='RandomFlipImgHHABefore', prob=0.5),  #5
            dict(
                type='ResizeImgHHA',
                img_size=256
            ),
            dict(
                type='RandomRotateImageHHA',
                prob=1.,
                angle=4,  # 整图90旋转
            ),
            # dict(
            #     type='RotateCircleImgAndHHA',
            #     resize=256,
            #     angle=rotation_num,
            #     grid=grid_num,
            #     local_or_global='local',
            #     save_augmented=False,
            #     log_dir=work_dir),
            # dict(type='RandomFlipImgHHA', prob=0.5),
            dict(type='NormalizeImgAndHHA', img_norm_cfg=img_norm_cfg),
            # dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle_SUNRGBDHHA'),
            dict(
                type='Collect',
                keys=[
                    'img_aug1', 'img_aug2', 'gt_semantic_seg_aug1',
                    'gt_semantic_seg_aug2', 'sunrgbd_rotation1',
                    'sunrgbd_rotation2',
                    'ori_img', 'ori_gt_semantic_seg'
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
optimizer = dict(type='SGD', lr=0.1, weight_decay=1e-4, momentum=0.9)  # 2卡时对应学习率0.1，故而8卡 学习率为0.4
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
    step=[19379*3*2, 29068*3*2])
runner = dict(type='IterBasedRunner', max_iters=32297*3*2)  # 8b8，此时应该是48445
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

