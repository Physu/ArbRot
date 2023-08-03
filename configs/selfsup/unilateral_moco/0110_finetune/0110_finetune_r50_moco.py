norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='Classification',
    pretrained='/mnt/disk7/lhy/Cascade/mmsegmentation-0.14.1/newback/unilateral_moco/0108_sunrgbd_rot_and_loc_unilateral/0110/latest.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN'),
        frozen_stages=4
    ),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=2048, num_classes=10),

    train_cfg=dict(),
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
    samples_per_gpu=256,
    workers_per_gpu=4,
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
# lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
lr_config = dict(
    policy='step',                        # 优化策略
    warmup='linear',                      # 初始的学习率增加的策略，linear为线性增加
    warmup_iters=500,                     # 在初始的500次迭代中学习率逐渐增加
    warmup_ratio=0.0001,                 # 起始的学习率
    step=[11720, 17579])                  # 在第0.6*total epch 和0.9* total epoch时降低学习率

runner = dict(type='IterBasedRunner', max_iters=19532)

checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=3000, metric='mIoU')
work_dir = 'newback/0113_finetune/cifar10/0113_restnet50'
gpu_ids = range(0, 1)
