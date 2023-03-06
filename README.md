# VoxelFormer

**a. Create a conda virtual environment and activate it.**
```shell
conda create --name VoxelFormer -y python=3.8
conda activate VoxelFormer
conda install -y pip
```

**b. Install PyTorch and torchvision.**
```shell
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch==1.7
```

**c. Install mmcv-full.**
```shell
pip install openmim
mim install mmcv-full==1.4.0
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.24.1
pip install mmsegmentation==0.20.2
```

**e. Install mmdet3d from source code.**
```shell
git clone  https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
pip install -e .
cd ..
```

**f. Install Dual-View Attn from source code.**
```shell
cd projects/mmdet3d_plugin/models/dv_attn
rm -rf build
python setup.py develop
cd ../../../..
```

**g. Install other requirements.**
```shell
pip install -r requirements.txt
```

**h. Prepare pretrained models.**
```shell
mkdir ckpts
```
