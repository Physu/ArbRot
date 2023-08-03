import pickle

file_name = "/data1/lhy/InfiRot/mmsegmentation/newback/ArbRotPlus/0519_ArbRotPolicy_res50_2b8_lr_1e-1_100e_scratch_byol_locm_rot_rgd_dgr_aug12_type2_cosinelr/policy.pickle"
with open(file_name, 'rb') as handle:
    policy = pickle.load(handle)

print(policy)