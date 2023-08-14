_base_ = f'/home/moe/MMaction/config/__base__/default_runtime.py'

chdir = "{{ fileDirname }}/.."

url = ('https://download.openmmlab.com/mmaction/recognition/slowfast/'
       'slowfast_r50_4x16x1_256e_kinetics400_rgb/'
       'slowfast_r50_4x16x1_256e_kinetics400_rgb_20200704-bcde7ed7.pth')

def load_label_map(file_path):
    """Load Label Map.

    Args:
        file_path (str): The file path of label map.

    Returns:
        dict: The label map (int -> label name).
    """
    # lines = open(file_path).readlines()
    # # lines = [x.strip().split(': ') for x in lines]
    # return {i+1: x for i, x in enumerate(lines)}
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    custom_classes = [int(x[0]) for x in lines]
    # [custom_classes.remove(i) for i in [1,2,3,4,5]]
    num_classes = len(custom_classes)  + 1
    return num_classes, custom_classes

label_map = f"{chdir}/download/KOKUYO_data/annotations/classes_en.txt" # label map file
num_classes, custom_classes = load_label_map(label_map)

model = dict(
    type='FastRCNN',
    _scope_='mmdet',
    init_cfg=dict(type='Pretrained', checkpoint=url),
    backbone=dict(
        type='mmaction.ResNet3dSlowFast',
        pretrained=None,
        resample_rate=8,
        speed_ratio=8,
        channel_ratio=8,
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            spatial_strides=(1, 2, 2, 1)),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            spatial_strides=(1, 2, 2, 1))),
    roi_head=dict(
        type='AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=4,
            with_temporal_pool=True),
        bbox_head=dict(
            type='BBoxHeadAVA',
            in_channels=2304,
            num_classes=num_classes,
            multilabel=True,
            dropout_ratio=0.5)),
    data_preprocessor=dict(
        type='mmaction.ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.9,
                neg_iou_thr=0.9,
                min_pos_iou=0.9),
            sampler=dict(
                type='RandomSampler',
                num=32,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0)),
    test_cfg=dict(rcnn=None))

dataset_type = 'AVADataset'
data_root = ""

ann_file_train = "" #f'{anno_root}/train.csv'
ann_file_val = ""#f'{anno_root}/train.csv'

exclude_file_train = None#f'{anno_root}/exclude.txt'
exclude_file_val = None#f'{anno_root}/exclude.txt'

label_file = ""#f'{anno_root}/action_list.pbtxt'#

proposal_file_train = (f'ava_data/annotations/ava_dense_proposals_train.FAIR.'
                       'recall_93.9.pkl')
proposal_file_val = f'ava_data/annotations/ava_dense_proposals_val.FAIR.recall_93.9.pkl'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=8, frame_interval=1),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]

# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(
        type='SampleAVAFrames', clip_len=8, frame_interval=1, test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]

test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        num_classes=num_classes,
        custom_classes=custom_classes,
        proposal_file=proposal_file_train,
        data_prefix=dict(img=data_root)))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        num_classes=num_classes,
        custom_classes=custom_classes,
        proposal_file=proposal_file_val,
        data_prefix=dict(img=data_root),
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='AVAMetric',
    ann_file=ann_file_val,
    label_file=label_file,
    num_classes=num_classes,
    custom_classes=custom_classes,
    exclude_file=exclude_file_val)
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=20, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        by_epoch=True,
        milestones=[10, 15],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.00001),
    clip_grad=dict(max_norm=40, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)