# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=120000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=120001, metric='mIoU')  # 暂不进行evaluation
# 一共5285张训练图片，batchsize=4，每个gpu放两张图片， 一共20 epoch