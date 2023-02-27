exp_id=debug

CUDA_VISIBLE_DEVICES=1 tools/dist_train.sh projects/configs/voxelformer/voxelformer_r50_3enc_800x320_D.py 1 28506 \
    --work-dir work_dirs/$exp_id \
    #--resume-from work_dirs/$exp_id/latest.pth