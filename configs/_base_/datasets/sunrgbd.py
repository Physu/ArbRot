# dataset settings
dataset_type = 'SUNRGBDDataset'
data_root = 'data/sunrgbd/sunrgbd_trainval'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_norm_cfg = dict(
    mean=[125.912, 116.549, 110.436], std=[71.267, 73.216, 74.424], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSUNRGBDAnnotations'),
    # dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    # dict(type='ResizeRotatePaste', resize=(225, 225), small_resize=(75, 75)), # ResizeRotatePaste 和 RotateCicle不可同时使用
    dict(type='RotateCircle', resize=225, angle=4, grid=4, local_or_global='global', radius_ratio=1.0),
    dict(type='Normalize', **img_norm_cfg),  # 这个normalize后面肯定需要
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle_SUNRGBD'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'sunrgbd_rotation', 'sunrgbd_location']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='image',
        ann_dir='depth_bfx_png',
        split='train_data_idx_backup.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='image',
        ann_dir='depth_bfx_png',
        split='val_data_idx_backup.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='image',
        ann_dir='depth_bfx_png',
        split='val_data_idx_backup.txt',
        pipeline=test_pipeline))
