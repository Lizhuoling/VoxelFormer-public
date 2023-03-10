_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin=True
plugin_dir='projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 0.8]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
time_with_grad_ = False
bev_h_ = 128
bev_w_ = 128
bev_z_ = 8
d_bound_ = [2.0, 58.0, 0.5]
T_ = 2
use_3d_=False
use_can_bus_ = False
loc_uncern_ = True

with_depth_ = True

x_bound = [-51.2, 51.2, 102.4 / bev_w_]
y_bound = [-51.2, 51.2, 102.4 / bev_h_]
z_bound = [-5, 3, 8 / bev_z_]

encoder_layer_num_ = 3
decoder_layer_num_ = 6

train_pkl_ = 'voxel_nuscenes_temporal_infos_train.pkl'
val_pkl_ = 'voxel_nuscenes_temporal_infos_val.pkl'
test_pkl_ = 'voxel_nuscenes_temporal_infos_test.pkl'

'''
train_pkl_ = 'voxel_nuscenes_temporal_infos_train_demo.pkl'
val_pkl_ = 'voxel_nuscenes_temporal_infos_val_demo.pkl'
test_pkl_ = 'voxel_nuscenes_temporal_infos_test_demo.pkl'
'''

model = dict(
    type='VoxelFormer',
    use_grid_mask=True,
    time_with_grad=time_with_grad_,
    pretrained=dict(img='torchvision://resnet50'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=512,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='VoxelFormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        bev_z=bev_z_,
        use_3d=use_3d_,
        num_classes=10,
        in_channels=512,
        with_depth=with_depth_,
        time_with_grad=time_with_grad_,
        d_bound=d_bound_,
        downsample_factor=32,
        num_query=900,
        with_time=True,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        loc_uncern=loc_uncern_,
        transformer=dict(
            type='VFTransformer',
            x_bound=x_bound,
            y_bound=y_bound,
            z_bound=z_bound,
            d_bound=d_bound_,
            use_3d=use_3d_,
            use_can_bus=use_can_bus_,
            time_with_grad=time_with_grad_,
            T=T_,
            encoder=dict(
                type='VFTransformerEncoder',
                num_layers=encoder_layer_num_,
                pc_range=point_cloud_range,
                num_points_in_pillar=bev_z_,
                return_intermediate=False,
                transformerlayers=dict(
                    type='VFTransformerEncoderLayer',
                    attn_cfgs=[
                        dict(
                            type='VFSelfAttention',
                            embed_dims=_dim_,
                            T=T_,
                            num_levels=1),
                        dict(
                            type='DVAttention',
                            embed_dims=_dim_,
                            use_3d=use_3d_,
                            bev_h=bev_h_,
                            bev_w=bev_w_,
                            bev_z=bev_z_,
                            num_heads=8,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=decoder_layer_num_,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],

                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10), 
        positional_encoding=dict(
            type='LearnedPositionalEncoding3D',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            hei_num_embed=bev_z_,
            use_3d=use_3d_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range))))

dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'

file_client_args = dict(backend='disk')

ida_aug_conf = {
        "resize_lim": (0.47, 0.625),
        "final_dim": (320, 800),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
    }

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiViewImageFromMultiSweepsFiles', sweeps_num=1, to_float32=True, pad_empty_sweeps=True, test_mode=False, sweep_range=[3,27], use_can_bus=use_can_bus_),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='LoadMultiViewImageDepthMap', num=6),
    dict(type='GlobalRotScaleTransImage',
            rot_range=[-0.3925, 0.3925],
            translation_std=[0, 0, 0],
            scale_ratio_range=[0.95, 1.05],
            reverse_angle=True,
            training=True
            ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_depth'],
            meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'timestamp', 'cur_can_bus', 'prev_can_bus'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiViewImageFromMultiSweepsFiles', sweeps_num=1, to_float32=True, pad_empty_sweeps=True, sweep_range=[3,27], use_can_bus=use_can_bus_),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'],
            meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'timestamp', 'cur_can_bus', 'prev_can_bus'))
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + train_pkl_,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, pipeline=test_pipeline, ann_file=data_root + val_pkl_, classes=class_names, modality=input_modality),
    test=dict(type=dataset_type, pipeline=test_pipeline, ann_file=data_root + val_pkl_, classes=class_names, modality=input_modality))


optimizer = dict(
    type='AdamW', 
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=1e-5)

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512., grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )
total_epochs = 24
evaluation = dict(interval=24, pipeline=test_pipeline)
find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=24, max_keep_ckpts=3)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# mAP: 0.2620
# mATE: 0.8719
# mASE: 0.2899
# mAOE: 0.6496
# mAVE: 0.5431
# mAAE: 0.2072
# NDS: 0.3748
# Eval time: 186.4s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.407   0.693   0.164   0.121   0.500   0.215
# truck   0.203   0.872   0.240   0.203   0.405   0.216
# bus     0.243   0.984   0.223   0.104   1.220   0.336
# trailer 0.055   1.198   0.281   0.771   0.276   0.045
# construction_vehicle    0.049   1.103   0.518   1.467   0.124   0.365
# pedestrian      0.342   0.796   0.300   0.799   0.605   0.237
# motorcycle      0.246   0.875   0.265   0.821   0.879   0.189
# bicycle 0.260   0.686   0.286   1.402   0.336   0.054
# traffic_cone    0.449   0.628   0.332   nan     nan     nan
# barrier 0.365   0.884   0.289   0.159   nan     nan