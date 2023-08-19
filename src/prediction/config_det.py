chdir = "/home/moe/MMaction"

clip_len = 8
frame_interval = 1
predict_stepsize = 4
label_map = f"{chdir}/config/coco-labels-2014_2017.txt"
det_config = f"{chdir}/mmaction2/demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py" 
det_checkpoint = f"{chdir}/download/model/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth" #"http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth" #human detection checkpoint file/url'
det_score_thr = 0.9 # the threshold of human detection score
det_cat_ids = [i for i in range(80)]
device = "cuda:0" # CPU/CUDA device option
output_fps = 4
output_stepsize = 4