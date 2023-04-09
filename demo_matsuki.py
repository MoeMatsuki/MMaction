from utils import load_yaml
import argparse
import torch
import pandas as pd
import pickle
import numpy as np
import cv2
import os.path as osp
import shutil
import os
from mmcv import Config

from preprocessor import Preprocessor
from detector import Detector
from recognizer import Recognizer
from analyzer import Analyzer

DET_PICKLE = None#"det_result/det_pickled.pkl"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', default='config/default.py')
    parser.add_argument('--video', default='KOKUYO_data/Webmtg_221226_01.MOV')
    parser.add_argument("--out_video", default="test_mmaction.mp4")
    parser.add_argument("--out_csv", default="test_mmaction.csv")
    args = parser.parse_args()
    return args


def dense_timestamps(timestamps, n):
    """Make it nx frames."""
    old_frame_interval = (timestamps[1] - timestamps[0])
    start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
    new_frame_inds = np.arange(
        len(timestamps) * n) * old_frame_interval / n + start
    return new_frame_inds.astype(np.int64)


args = parse_args()
conf = Config.fromfile(args.conf_path)
conf.merge_from_dict(conf.cfg_options)
# conf = load_yaml(args.conf_path)

pre_processing = Preprocessor(conf)
frame_paths = pre_processing(args.video)

if DET_PICKLE is None:
    detector = Detector(conf)
    human_detections = detector(pre_processing)
    # pickle化してファイルに書き込み
    # with open(DET_PICKLE, 'wb') as f:
    #     pickle.dump(human_detections, f)
else:
    with open(DET_PICKLE, 'rb') as f:
        human_detections = pickle.load(f)

recognizer = Recognizer(conf, pre_processing)
results = recognizer(human_detections)

dense_n = int(conf.predict_stepsize / conf.output_stepsize)
pre_processing.frames = [
    cv2.imread(pre_processing.frame_paths[i - 1])
    for i in dense_timestamps(pre_processing.timestamps, dense_n)
]
pre_processing.frame_paths = [
    pre_processing.frame_paths[i - 1]
    for i in dense_timestamps(pre_processing.timestamps, dense_n)
]
print('Performing visualization')   
print(results)
labels = pre_processing.get_label()
label_names = list(labels.values())[5:]
analyzer = Analyzer(label_names)
df_result, vid = analyzer(conf, pre_processing.frame_paths, pre_processing.frames, results)
pd.DataFrame(df_result).to_csv(args.out_csv)

vid.write_videofile(args.out_video)

tmp_frame_dir = osp.dirname(pre_processing.frame_paths[0])
shutil.rmtree(tmp_frame_dir)

