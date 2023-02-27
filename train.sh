exp_id=voxelformer_r50_800x320_TD

CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_train.sh projects/configs/voxelformer/voxelformer_r50_800x320_TD.py 4 28506 \
    --work-dir work_dirs/$exp_id \
