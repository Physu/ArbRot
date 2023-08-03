#python tools/train.py configs/selfsup/moco/0710_custome/125_with_pretrain/0715_lr_1e-1_200e_b32_moco.py --seed 0
#sleep 15
#python tools/train.py configs/selfsup/moco/0710_custome/125_with_pretrain/0715_lr_1e-2_200e_b32_moco.py --seed 0
#sleep 15
#python tools/train.py configs/selfsup/moco/0710_custome/125_with_pretrain/0715_lr_1e-3_200e_b32_moco.py --seed 0

#python tools/train.py configs/selfsup/moco/0710_custome/125_with_pretrain/0717_lr_1e-1_200e_b32_dgr.py --seed 0
#sleep 15
#python tools/train.py configs/selfsup/moco/0710_custome/125_with_pretrain/0717_lr_1e-1_200e_b32_moco_dgr.py --seed 0
#sleep 15
#python tools/train.py configs/selfsup/moco/0710_custome/125_with_pretrain/0717_lr_1e-1_200e_b32_moco_loc_rot.py --seed 0
#sleep 15
#python tools/train.py configs/selfsup/moco/0710_custome/125_with_pretrain/0717_lr_1e-1_200e_b32_moco_loc_rot_rgd_dgr.py --seed 0
#sleep 15
#python tools/train.py configs/selfsup/moco/0710_custome/125_with_pretrain/0717_lr_1e-1_200e_b32_moco_rgd.py --seed 0
#sleep 15
#python tools/train.py configs/selfsup/moco/0710_custome/125_with_pretrain/0717_lr_1e-1_200e_b32_moco_rgd_rgd.py --seed 0
#sleep 15
#python tools/train.py configs/selfsup/moco/0710_custome/125_with_pretrain/0717_lr_1e-1_200e_b32_moco_rot_dgr.py --seed 0


#python tools/train.py configs/selfsup/moco/0710_custome/125_with_pretrain/0721_lr_1e-1_200e_b32_moco_loc_dgr.py --seed 0
#sleep 15
#python tools/train.py configs/selfsup/moco/0710_custome/125_with_pretrain/0721_lr_1e-1_200e_b32_moco_loc_rgd.py --seed 0
#sleep 15
#python tools/train.py configs/selfsup/moco/0710_custome/125_with_pretrain/0721_lr_1e-1_200e_b32_moco_loc_rot_dgr.py --seed 0
#sleep 15
#python tools/train.py configs/selfsup/moco/0710_custome/125_with_pretrain/0721_lr_1e-1_200e_b32_moco_loc_rot_rgd.py --seed 0
#sleep 15
#python tools/train.py configs/selfsup/moco/0710_custome/125_with_pretrain/0721_lr_1e-1_200e_b32_moco_rgd.py --seed 0
#sleep 15
#python tools/train.py configs/selfsup/moco/0710_custome/125_with_pretrain/0721_lr_1e-1_200e_b32_moco_rot.py --seed 0


#python tools/train.py configs/selfsup/moco/0710_custome/0720_lr_1e-1_200e_b32_aug12345.py --seed 0
#sleep 15
#
#python tools/train.py configs/selfsup/moco/0710_custome/0720_lr_1e-1_200e_b32_aug12.py --seed 0
#
#python tools/train.py configs/selfsup/moco/augmentation-ablation/0720_res50_b16_lr_1e-1_100e_all_reweight_grid3.py --seed 0
#sleep 15
#python tools/train.py configs/selfsup/moco/augmentation-ablation/0720_res50_b16_lr_1e-1_100e_all_reweight_grid4.py --seed 0

#tools/dist_train_moco1.sh configs/ablation/rot/0801_res50_b16_lr_1e-1_200e_360rot_center_paste.py 2 --no-validate --seed 2022
#sleep 3600
export CUDA_VISIBLE_DEVICES=6,7

tools/dist_train_moco1.sh configs/ArbRotPlus/Policy/0621_ArbRotPlusPolicy_res50_2b8_lr_1e-1_100e_scratch_byol_locm_rot_rgd_dgr_aug12_type2_cosinelr.py 2 --no-validate \
--load-from /data1/lhy/InfiRot/mmsegmentation/newback/ArbRotPlus/0603_ArbRotPlus_res50_2b8_lr_1e-1_300e_scratch_byol_locm_circle_rot_rgd_dgr_aug12_retrain_cosinelr/iter_99096.pth
#sleep 10
#tools/dist_train_moco1.sh configs/selfsup/moco/from_scratch/20230619_marc_res50_2b8_lr_1e-1_300e_arbrot_local.py 2 --no-validate
#--resume /data1/lhy/InfiRot/mmsegmentation/newback/ArbRotPlus/0606_ArbRotPlus_res50_2b8_lr_1e-1_300e_scratch_simsiam_locm_circle_rot_rgd_dgr_aug12_retrain_cosinelr/iter_49548.pth
