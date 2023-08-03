# 主要用于simplehead部分的代码测试
# 不包括复杂的头部结构，只用简单的网络实现对比学习和其他pretask 任务
# 但是包括了rgb和depth两个输入分支，用来分别完成对应的任务
# 0311这个主要是对于图片恢复和深度估计部分代码，进行了修正 sigmoid0
rotation_num = 90
num_rot_classes = 12
grid_num = 9
# norm_cfg = dict(type='BN', requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
dataset_type = 'SUNRGBDHHADataset'
data_root = 'data/sunrgbd/sunrgbd_trainval'

# model settings
model = dict(
    type='MoCoUnet0321',
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
        in_channels=512,
        hid_channels=512,
        out_channels=128,
        with_avg_pool=True),

    contrastive_head=dict(type='ContrastiveHead', temperature=0.2)
)

data = dict(
    # samples_per_gpu=32,
    # workers_per_gpu=4,  # 关于这个worker设置：4*gpu_num https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/3
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='image',
        # ann_dir='depth_bfx_png',
        ann_dir='hha_bfx',
        split='trainval_data_idx_backup.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadSUNRGBDAnnotations', keep_origin=True),  # keep_origin 是否保持深度信息，还是uint8形式
            # dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
            # dict(type='RandomFlip', prob=0.5),
            # 不要角度的影响，只考虑位置的影响
            dict(type='RotateCircleImgAndHHA',  # 包括了resize功能
                 resize=256,
                 angle=rotation_num,
                 grid=grid_num,  # 注意 resize//grid 的结果表示分割的距离
                 local_or_global='global',  # option:'local' or 'global'
                 save_augmented=False),
            dict(type='RandomFlipImgHHA', prob=0.5),
            # dict(type='PhotoMetricDistortion'),
            dict(
                type='NormalizeImgAndHHA'),
            dict(type='DefaultFormatBundle_SUNRGBDHHA'),
            dict(
                type='Collect',
                # 虽然关键词里没有img，但是最后输入到backbone里面还是有img这个关键词，具体实现在mmseg/datasets/sunrgbd_moco.py 中的 prepare_train_img 方法
                keys=[  # mmseg/datasets/pipelines/formating_sunrgbd.py
                    'img_aug1', 'img_aug2',
                    'gt_semantic_seg_aug1', 'gt_semantic_seg_aug2',
                    'sunrgbd_rotation1', 'sunrgbd_rotation2',
                    'paste_location1', 'paste_location2',
                ])
        ],
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
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
        type=dataset_type,
        data_root=data_root,
        img_dir='image',
        ann_dir='hha_bfx',
        split='val_data_idx_backup.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))

# optimizer
optimizer = dict(type='SGD', lr=0.00001, weight_decay=0.0001, momentum=0.9)
optimizer_config = dict()
runner = dict(type='IterBasedRunner', max_iters=32297*4)  # train 10335*100(epoch) /16*2gpu
# runner = dict(type='IterBasedRunner', max_iters=66075)  # 200 epoch
# learning policy
# lr_config = dict(policy='CosineAnnealing', min_lr=0.)
lr_config = dict(
    policy='step',  # 优化策略
    warmup='linear',  # 初始的学习率增加的策略，linear为线性增加
    warmup_iters=500,  # 在初始的500次迭代中学习率逐渐增加
    warmup_ratio=0.0001,  # 起始的学习率
    step=[19379*4, 29068*4])  # 在第0.6*total epch 和0.9* total epoch时降低学习率

checkpoint_config = dict(interval=60000)
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
work_dir = 'newback/moco/resnext101_new/0402_resnext101_32x8d_bsize_8_lr_1e-5_rotandloc.py'
