# _base_ = "slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py" # spatio temporal detection config file path
_base_ = "slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes_n.py"

chdir = "{{ fileDirname }}/.."

# checkpoint = f"/home/moe/MMaction/checkpoints/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb_20220906-d934a48f.pth" #spatio temporal detection checkpoint file/url"
checkpoint = f"{chdir}/download/model/latest_20230614_slowfast_epoch392" 
# checkpoint = f"{chdir}/work_dirs/slowfast/20230614_fastlabel_soloweb/epoch_10.pth"
det_config = f"{chdir}/mmaction2/demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py" #human detection config file path (from mmdet)
det_checkpoint = f"{chdir}/download/model/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth" #"http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth" #human detection checkpoint file/url'
det_score_thr = 0.9 # the threshold of human detection score
action_score_thr = 0.5 # the threshold of human action score
label_map = f"{chdir}/fastlabel1/annotations/classes_en.txt" # label map file
device = "cuda:0" # CPU/CUDA device option
predict_stepsize = 4 # give out a prediction per n frames
output_stepsize = 4 # show one frame per n frames in the demo, we should have predict_stepsize % output_stepsize == 0
output_fps = predict_stepsize # the fps of demo video output
work_dir = ('work_dirs/slowfast/20230619_fastlabel') # override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. For example, '--cfg-options model.backbone.depth=18 model.backbone.with_cp=True

 # override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. For example, '--cfg-options model.backbone.depth=18 model.backbone.with_cp=True
import os
JSON_DIR = 'fastlabel1/json'
TRAIN_IMG_DIR = 'fastlabel1/train_img'
TRAIN_TXT_DIR = 'fastlabel1/annotation'
data_root = "fastlabel1/convert_img"
anno_root = "fastlabel1/annotations"
# TRAIN_IMG_DIR = 'download/KOKUYO_data/train_data'#'fastlabel1/train_img'
# TRAIN_TXT_DIR = 'download/KOKUYO_data/train_data'#'fastlabel1/annotation'
# data_root = 'download/KOKUYO_data/convert_img'#"fastlabel1/convert_img"
# anno_root = 'download/KOKUYO_data/annotations'#"fastlabel1/annotations"
label_file = f'{anno_root}/action_list.pbtxt'
action_label = f"{anno_root}/classes_en2.txt"
action_label_all = f"{anno_root}/classes_en3.txt"
anno_train_csv = os.path.join(anno_root, "train.csv")
anno_val_csv = os.path.join(anno_root, "val.csv")
rf_train_csv = os.path.join(anno_root, "train_rf.csv")
rf_val_csv = os.path.join(anno_root, "val_rf.csv")
val_videos  = ["IMG_1936_5", "IMG_1936_10", "IMG_1811", "IMG_0598", "IMG_0575"]
det_cat_id = 0

rf_model = "/home/moe/MMaction/config/clf_model_WB.pkl"
rf_pickle_path = os.path.join(work_dir, rf_model)
true_ids = [3,4]#[59, 64, 67, 104]
exclude_sample_ids = []#[1, 123]
flag_multi_cls = False
train_epoch = 1000
result_rf_path = "result_rf.csv"
frm_thresh = 40

cfg_options = {
    "train_dataloader.dataset.ann_file": anno_train_csv,
    "val_dataloader.dataset.ann_file": anno_val_csv,
    "train_dataloader.dataset.data_prefix": dict(img=data_root),
    "val_dataloader.dataset.data_prefix": dict(img=data_root),
    "train_dataloader.dataset.label_file": label_file,
    "val_dataloader.dataset.label_file": label_file,
    "val_evaluator.ann_file": anno_val_csv,
    "val_evaluator.label_file": label_file,
    "train_cfg.max_epochs": train_epoch,
    "load_from": checkpoint
}