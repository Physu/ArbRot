# model settings
# norm_cfg = dict(type='SyncBN', requires_grad=True)
# 在configs/_base_/models 的文件中，首行norm_cfg = dict(type='SyncBN', requires_grad=True), 'SyncBN'是采用distributed的训练方法，在单GPU non-distributed训练中使用会出现上述错误，
# 原文链接：https://blog.csdn.net/m0_37568067/article/details/109785209
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoderSUNRGBD',  # 注意这里修改了
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),  # 需要输出的 stage 的索引
        dilations=(1, 1, 2, 4),  # 与前面的stage相对应的，每个stage第一个bottleneck正常，余下的bottleneck中conv3*3就是这个对应的dilation
        strides=(1, 2, 1, 1),  # 对应每个stage第一个bottleneck的conv3*3 和后面downsample 步长
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='ASPPHead',
        in_channels=2048,
        in_index=3,  # 这个对应最后选取的stage的输出，这里选取的是第四个stage
        channels=512,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=19,  # change with first config
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='MSELoss')
        # loss_decode=dict(
        #     type='SmoothL1Loss', beta=1.0, loss_weight=0.4),
    ),
    auxiliary_head=dict(
        type='FCNHeadSUNRGBD',
        in_channels=(1024, 2048),
        input_transform='multiple_select',  # 注意此处，为下面多stage结果输出做准备
        in_index=(2, 3),  # 这个对应最后选取的stage的输出，这里选取的是第三个stage
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=dict(
        #     type='SmoothL1Loss', beta=1.0, loss_weight=0.4),
        loss_decode=dict(
            type="MSELoss"),
        loss_rotation=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
        loss_location=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
