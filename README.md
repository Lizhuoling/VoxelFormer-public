# VoxelFormer

This is the official implementation of the paper "VoxelFormer: Bird’s-Eye-View Feature Generation based on Dual-view Attention for Multi-view 3D Object Detection"

## Install

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
Download the pre-trained resnet101 and vovnet in this folder. You can download them from [resnet101](https://cloud.tsinghua.edu.cn/f/73197126b0f44d819112/?dl=1) and [vovnet](https://cloud.tsinghua.edu.cn/f/24666d097bc64f71ac7d/?dl=1).

## Nuscenes
Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download). Prepare nuscenes data by running

**Prepare nuScenes data**

```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag voxel_nuscenes
```
Using the above code will generate `voxel_nuscenes_infos_{train,val}.pkl`.

```
python tools/voxel_generate_sweep_pkl.py --split train
python tools/voxel_generate_sweep_pkl.py --split val
# python tools/voxel_generate_sweep_pkl.py --split test
```
Using the above code will generate `voxel_nuscenes_temporal_infos_{train,val}.pkl`.

(optional)
```
python tools/voxel_generate_demo_pkl.py --split train
python tools/voxel_generate_demo_pkl.py --split val
# python tools/voxel_generate_demo_pkl.py --split test
```
Using the above code will generate `voxel_nuscenes_temporal_infos_{train,val}_demo.pkl`.

**Folder structure**
```
VoxelFormer
├── projects/
├── tools/
├── configs/
├── ckpts/
│   ├── fcos3d-res101.pth
│   ├── fcos3d_vovnet_imgbackbone-remapped.pth
├── mmdetection3d/
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── voxel_nuscenes_infos_trian.pkl
|   |   ├── voxel_nuscenes_infos_val.pkl
|   |   ├── voxel_nuscenes_temporal_infos_trian.pkl
|   |   ├── voxel_nuscenes_temporal_infos_val.pkl
```

**Train**
We provide four config files for training in `projects/configs/voxelformer/`. You can train a model by:
```
bash tools/dist_train.sh $config_path $gpu_num $port --work-dir work_dirs/$exp_save_path
```
For inference, use:
```
bash tools/dist_test.sh $config_path $path_to_ckpt $gpu_nm --eval bbox
```
