_base_ = '/home/moe/MMaction/config/__base__/default_runtime.py'

cdir = "{{ fileDirname }}/.."


url = ('https://download.openmmlab.com/mmaction/recognition/slowonly/'
       'omni/slowonly_r101_without_omni_8x8x1_kinetics400_rgb_'
       '20200926-0c730aef.pth')

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
    [custom_classes.remove(i) for i in [1,2,3,4,5]]
    num_classes = len(custom_classes) + 1
    return num_classes

label_map = f"{cdir}/download/KOKUYO_data/annotations/classes_en.txt" # label map file
num_classes = load_label_map(label_map)

# model setting
model = dict(
    type='FastRCNN',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=101,
        pretrained=None,
        pretrained2d=False,
        lateral=False,
        num_stages=4,
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        spatial_strides=(1, 2, 2, 1)),
    roi_head=dict(
        type='AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True),
        bbox_head=dict(
            type='BBoxHeadAVA',
            in_channels=2048,
            num_classes=num_classes,
            multilabel=True,
            dropout_ratio=0.5)),
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
            pos_weight=1.0,
            debug=False)),
    test_cfg=dict(rcnn=dict(action_thr=0.002)))

dataset_type = 'AVADataset'
data_root = ""

ann_file_train = "" #f'{anno_root}/train.csv'
ann_file_val = ""#f'{anno_root}/train.csv'

exclude_file_train = None#f'{anno_root}/exclude.txt'
exclude_file_val = None#f'{anno_root}/exclude.txt'

label_file = ""#f'{anno_root}/action_list.pbtxt'#
clip_len = 4 #8
total_epochs = 20

proposal_file_train = (f'ava_data/annotations/ava_dense_proposals_train.FAIR.'
                       'recall_93.9.pkl')
proposal_file_val = f'ava_data/annotations/ava_dense_proposals_val.FAIR.recall_93.9.pkl'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=clip_len, frame_interval=8),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    # Rename is needed to use mmdet detectors
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img', 'proposals', 'gt_bboxes', 'gt_labels']),
    dict(
        type='ToDataContainer',
        fields=[
            dict(key=['proposals', 'gt_bboxes', 'gt_labels'], stack=False)
        ]),
    dict(
        type='Collect',
        keys=['img', 'proposals', 'gt_bboxes', 'gt_labels'],
        meta_keys=['scores', 'entity_ids'])
]
# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(type='SampleAVAFrames', clip_len=clip_len, frame_interval=8, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    # Rename is needed to use mmdet detectors
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img', 'proposals']),
    dict(type='ToDataContainer', fields=[dict(key='proposals', stack=False)]),
    dict(
        type='Collect',
        keys=['img', 'proposals'],
        meta_keys=['scores', 'img_shape'],
        nested=True)
]
data = dict(
    videos_per_gpu=6,
    workers_per_gpu=2,
    # During testing, each video may have different shape
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_train,
        person_det_score_thr=0.9,
        num_classes=num_classes,
        custom_classes=num_classes,
        filename_tmpl="img_{:05d}.jpg",
        data_prefix=data_root),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        person_det_score_thr=0.9,
        num_classes=num_classes,
        custom_classes=num_classes,
        filename_tmpl="img_{:05d}.jpg",
        data_prefix=data_root))
data['test'] = data['val']

optimizer = dict(type='SGD', lr=0.075, momentum=0.9, weight_decay=0.00001)
# this lr is used for 8 gpus

optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy

lr_config = dict(
    policy='step',
    step=[10, 15],
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5,
    warmup_ratio=0.1)
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]

evaluation = dict(interval=1, save_best='mAP@0.5IOU')
log_config = dict(
    interval=1, hooks=[
        dict(type='TextLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = ('./checkpoints/'
#             'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth')
load_from = ('https://download.openmmlab.com/mmaction/recognition/slowonly/'
             'omni/' 
             'slowonly_r101_omni_8x8x1_kinetics400_rgb_20200926-b5dbb701.pth')

resume_from = None
find_unused_parameters = False
