_base_ = [
    '../_base_/models/deeplabv3_r50-d8_sunrgbd.py',
    '../_base_/datasets/sunrgbd.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(num_classes=1), auxiliary_head=dict(num_classes=1))  # 只需要预测深度，其他的不需要，loss也为smoothL1
