rotation_num = None
num_rot_classes = 12
grid_num = 25
loc_specify = None  # 这个和grid_num相关，取grid中间的位置
norm_cfg = dict(type='BN', requires_grad=True)
# norm_cfg = dict(type='SyncBN', requires_grad=True)
dataset_type = 'SUNRGBDHHADataset'
data_root = 'data/sunrgbd/sunrgbd_trainval'
work_dir = 'newback/moco/from_scratch/rot_ablation/0802_res50_b16_lr_1e-1_200e_360rot_random_paste'

# 0406 1.用了ssim 2.对于rot res 预测，smoothl1loss改为l1 loss
# 0411 1.将单纯的0-1.0，通过imagenet给归一化到ImageNet那个范围
# 0412 注意这里，重新修改了程序
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    to_rgb=True)
label_norm_cfg = dict(mean=[19050.278], std=[9693.823])
model = dict(
    type='MoCoUnet0801',
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
        bias=True,
        loss_total_weight=1.0)
)
data = dict(
    samples_per_gpu=32,
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
            #             #      crop_size=(500, 500),
            #             #      padding=0),
            #             # dict(type='ColorJitterImgHHA',  #2
            #             #      brightness=0.4,
            #             #      contrast=0.4,
            #             #      saturation=0.4),
            # dict(type='RandomGrayscaleImgHHA',  #3
            #      gray_prob=0.1),
            # dict(type='GaussianBlurImgHHA', sigma_min=0.1, sigma_max=2.0, p=0.5),  #4
            # dict(type='RandomFlipImgHHABefore', prob=0.5),  #5
            dict(
                type='RotateCircleImgAndHHA',
                resize=256,
                angle=rotation_num,
                grid=grid_num,
                loc_specify=loc_specify,
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
optimizer = dict(type='SGD', lr=1., weight_decay=1e-4, momentum=0.9)
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
    # step=[19379*2, 29068*2],
    step=[19379, 29068]
)
# runner = dict(type='IterBasedRunner', max_iters=32297*2)
runner = dict(type='IterBasedRunner', max_iters=32297)
checkpoint_config = dict(interval=32297)

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

# evaluation = dict(interval=10,
#                   metric='mIoU',
#                   # save_best='auto',
#                   # rule='acc',
#                   # greater_keys=['img_rot_cls_acc', 'img_rot_res_gap', 'img_rot_gap', 'img_loc_acc',
#                   #               'depth_rot_cls_acc', 'depth_rot_res_gap', 'depth_rot_gap', 'depth_loc_acc',
#                   #               'rot_cls_acc', 'rot_res_gap', 'rot_gap', 'loc_acc']
#                 )  # evaluation 调用部分