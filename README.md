# VoxelFormer

conda create --name VoxelFormer -y python=3.8
conda activate VoxelFormer
conda install -y pip

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

pip install openmim
mim install mmcv-full==1.4.0

pip install mmdet==2.24.1
pip install mmsegmentation==0.20.2

git clone  https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
pip install -e .
cd ..

cd projects/mmdet3d_plugin/models/voxel_attn
rm -rf build
python setup.py develop
cd ../../../..

mkdir ckpts
