rotation_num = None
num_rot_classes = 12
grid_num = 9
norm_cfg = dict(type='BN', requires_grad=True)
dataset_type = 'SUNRGBDHHADataset'
data_root = 'data/sunrgbd/sunrgbd_trainval'
work_dir = 'newback/moco/0404_marcl_resnet50_bsize_16_lr_1e-3'

label_norm_cfg = dict(mean=[19050.278], std=[9693.823])
# model settings
model = dict(
    type='MoCo',
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='ResNet',
        depth=50,
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
    samples_per_gpu=16,
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
optimizer = dict(type='SGD', lr=0.001, weight_decay=0.0001, momentum=0.9)
optimizer_config = dict()
runner = dict(type='IterBasedRunner', max_iters=32297)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.0001,
    step=[19379, 29068])
checkpoint_config = dict(interval=10000)
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
work_dir = 'newback/moco/ablation/0404_marcl_resnet18_bsize_16_lr_1e-3_contrasitve'
