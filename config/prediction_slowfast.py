# _base_ = "slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py" # spatio temporal detection config file path
_base_ = "slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes.py"

chdir = "{{ fileDirname }}/.."

dataset_type = 'AVADataset'
data_root = 'KOKUYO_data/convert'
anno_root = 'KOKUYO_data/annotations'

ann_file_train = f'{anno_root}/train.csv'
ann_file_val = f'{anno_root}/train.csv'

exclude_file_train = f'{anno_root}/exclude.txt'
exclude_file_val = f'{anno_root}/exclude.txt'

label_file = f'{anno_root}/action_list.pbtxt'#

# checkpoint = f"{chdir}/work_dirs_kokuyo2/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/latest.pth" #spatio temporal detection checkpoint file/url
checkpoint = f"{chdir}/checkpoints/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb_20220906-d934a48f.pth" 
det_config = f"{chdir}/mmaction2/demo/faster_rcnn_r50_fpn_2x_coco.py" #human detection config file path (from mmdet)
det_checkpoint = f"{chdir}/checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth" #"http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth" #human detection checkpoint file/url'
det_score_thr = 0.9 # the threshold of human detection score
action_score_thr = 0.5 # the threshold of human action score
label_map = f"{chdir}/KOKUYO_data/annotations/classes_en.txt" # label map file
device = "cuda:0" # CPU/CUDA device option
predict_stepsize = 8 # give out a prediction per n frames
output_stepsize = 4 # show one frame per n frames in the demo, we should have predict_stepsize % output_stepsize == 0
output_fps = 6 # the fps of demo video output
cfg_options = {} # override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. For example, '--cfg-options model.backbone.depth=18 model.backbone.with_cp=True