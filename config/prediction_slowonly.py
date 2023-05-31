_base_ = "slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py" # spatio temporal detection config file path

cdir = "{{ fileDirname }}/.."

checkpoint = f"{cdir}/work_dirs/slowonly/20230529_fastlabel/latest.pth" #spatio temporal detection checkpoint file/url
det_config = f"{cdir}/mmaction2/demo/faster_rcnn_r50_fpn_2x_coco.py" #human detection config file path (from mmdet)
det_checkpoint = f"{cdir}/download/model/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth" #"http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth" #human detection checkpoint file/url'
det_score_thr = 0.9 # the threshold of human detection score
action_score_thr = 0.5 # the threshold of human action score
label_map = f"{cdir}/download/KOKUYO_data/annotations/classes_en.txt" # label map file
device = "cuda:0" # CPU/CUDA device option
predict_stepsize = 8 # give out a prediction per n frames
output_stepsize = 4 # show one frame per n frames in the demo, we should have predict_stepsize % output_stepsize == 0
output_fps = 6 # the fps of demo video output
work_dir = ('work_dirs/slowonly/20230601_fastlabel')

 # override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. For example, '--cfg-options model.backbone.depth=18 model.backbone.with_cp=True
import os
TRAIN_IMG_DIR = 'download/KOKUYO_data/train_data'#'fastlabel1/train_img'
TRAIN_TXT_DIR = 'download/KOKUYO_data/train_data'#'fastlabel1/annotation'
data_root = 'download/KOKUYO_data/convert_img'#"fastlabel1/convert_img"
anno_root = 'download/KOKUYO_data/annotations'#"fastlabel1/annotations"
label_file = f'{anno_root}/action_list.pbtxt'
action_label = f"{anno_root}/classes_en2.txt"
anno_train_csv = os.path.join(anno_root, "train.csv")
anno_val_csv = os.path.join(anno_root, "val.csv")
rf_train_csv = os.path.join(anno_root, "test_rf.csv")
rf_val_csv = os.path.join(anno_root, "val_rf.csv")
val_videos = ["IMG_1802", "IMG_0591", "IMG_1800"]

rf_model = "clf_model_WB.pkl"
rf_pickle_path = os.path.join(work_dir, rf_model)
true_ids = [59, 64, 67, 104]
exclude_sample_ids = [1, 123]
multi_cls = False

cfg_options = {
    "data.train.ann_file": anno_train_csv,
    "data.val.ann_file": anno_val_csv,
    "data.train.data_prefix": data_root,
    "data.val.data_prefix": data_root,
    "data.train.label_file": label_file,
    "data.val.label_file": label_file,
}