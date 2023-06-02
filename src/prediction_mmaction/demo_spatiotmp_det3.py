# Copyright (c) OpenMMLab. All rights reserved.
import setup as setup
import argparse
import copy as cp
import os
import os.path as osp
import shutil
import pandas as pd
from mmengine.config import Config
import glob

import cv2
import mmcv
import numpy as np
import torch
from natsort import natsorted
from mmcv import DictAction
from mmcv.runner import load_checkpoint

from mmaction.models import build_detector

from detector2 import Detector
from recognizer import Recognizer
from rendering import visualize

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this demo! ')

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('conf_path')
    parser.add_argument('--video', default='KOKUYO_data/Webmtg_221226_01.MOV')
    parser.add_argument('--out', default='.')
    parser.add_argument("--out_video", default="test_mmaction.mp4")
    parser.add_argument("--out_csv", default="test_mmaction.csv")
    args = parser.parse_args()
    return args

def frame_extraction(video_path):
    """Extract frames given video_path.
    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('tmp', osp.basename(osp.splitext(video_path)[0]))
    frame_paths = glob.glob(os.path.join(target_dir, '*.jpg'))
    frame_paths = natsorted(frame_paths)
    print(frame_paths)
    frames = [cv2.imread(i) for i in frame_paths]
    # os.makedirs(target_dir, exist_ok=True)
    # # Should be able to handle videos up to several hours
    # frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    # print(f"target video is {video_path}")
    # vid = cv2.VideoCapture(video_path)
    # frames = []
    # frame_paths = []
    # flag, frame = vid.read()
    # cnt = 0
    # while flag:
    #     frames.append(frame)
    #     frame_path = frame_tmpl.format(cnt + 1)
    #     frame_paths.append(frame_path)
    #     cv2.imwrite(frame_path, frame)
    #     cnt += 1
    #     flag, frame = vid.read()
    return frame_paths, frames

def save_csv(frames_path, frames, annotations, labels):

        anno = None
        frame_result = {
                    "frame": [],
                    "min_x": [],
                    "min_y": [],
                    "max_x": [],
                    "max_y": [],
                }
        [frame_result.update({l: []}) for l in labels]
        frames_ = cp.deepcopy(frames)
        nf, na = len(frames), len(annotations)
        assert nf % na == 0
        nfpa = len(frames) // len(annotations)
        anno = None
        h, w, _ = frames[0].shape
        scale_ratio = np.array([w, h, w, h])
        for i in range(na):
            anno = annotations[i]
            if anno is None:
                continue
            for j in range(nfpa):
                ind = i * nfpa + j
                for ann in anno:
                    box = ann[0]
                    label = ann[1]
                    if not len(label):
                        continue
                    score = ann[2]
                    box = (box * scale_ratio).astype(np.int64)
                    frame_result["frame"].append(frames_path[ind])
                    frame_result["min_x"].append(box[0])
                    frame_result["min_y"].append(box[1])
                    frame_result["max_x"].append(box[2])
                    frame_result["max_y"].append(box[3])
                    for lb in labels:
                        if lb in label:
                            k = label.index(lb)
                            frame_result[lb].append(str(score[k]))
                        else:
                            frame_result[lb].append(str(0))

        return frame_result

def main():
    args = parse_args()

    # save frames in tmp directory
    frame_paths, original_frames = frame_extraction(args.video)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape
    print("done extract frames")

    # resize frames to shortside 256
    new_w, new_h = mmcv.rescale_size((w, h), (256, np.Inf))
    frames = [mmcv.imresize(img, (new_w, new_h)) for img in original_frames]
    dirname = osp.dirname(os.path.join("tmp_resize", frame_paths[0]))
    os.makedirs(dirname, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join("tmp_resize", frame_paths[i]), f)
    w_ratio, h_ratio = new_w / w, new_h / h
    # print("done resize frames")

    # Get clip_len, frame_interval and calculate center index of each clip
    config = mmcv.Config.fromfile(args.conf_path)
    config.merge_from_dict(config["cfg_options"])
    val_pipeline = config.data.val.pipeline

    sampler = [x for x in val_pipeline if x['type'] == 'SampleAVAFrames'][0]
    clip_len, frame_interval = sampler['clip_len'], sampler['frame_interval']
    window_size = clip_len * frame_interval
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'
    # Note that it's 1 based here
    timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
                           config["predict_stepsize"])

    # Get Human detection results
    detector = Detector(config)
    human_detections, all_detection = detector(frame_paths, timestamps, w_ratio, h_ratio)
    detector.save_results(all_detection, frames)

    # Get img_norm_cfg
    img_norm_cfg = config['img_norm_cfg']
    if 'to_rgb' not in img_norm_cfg and 'to_bgr' in img_norm_cfg:
        to_bgr = img_norm_cfg.pop('to_bgr')
        img_norm_cfg['to_rgb'] = to_bgr
    img_norm_cfg['mean'] = np.array(img_norm_cfg['mean'])
    img_norm_cfg['std'] = np.array(img_norm_cfg['std'])

    # Build STDET model
    try:
        # In our spatiotemporal detection demo, different actions should have
        # the same number of bboxes.
        config['model']['test_cfg']['rcnn']['action_thr'] = .0
    except KeyError:
        pass

    recognizer = Recognizer(config)
    # print(human_detections[0].size(), type(all_detection))
    results = recognizer(human_detections, timestamps, clip_len, frame_interval, frames, new_h, new_w)

    def dense_timestamps(timestamps, n):
        """Make it nx frames."""
        old_frame_interval = (timestamps[1] - timestamps[0])
        start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
        new_frame_inds = np.arange(
            len(timestamps) * n) * old_frame_interval / n + start
        return new_frame_inds.astype(np.int64)

    dense_n = int(config["predict_stepsize"] / config["output_stepsize"])
    frames = [
        cv2.imread(frame_paths[i - 1])
        for i in dense_timestamps(timestamps, dense_n)
    ]
    frame_paths = [
        frame_paths[i - 1]
        for i in dense_timestamps(timestamps, dense_n)
    ]
    print('Performing visualization')
    vis_frames = visualize(frames, results)
    label_names = recognizer.get_label()
    labels = list(label_names.values())
    df_result = save_csv(frame_paths, frames, results, labels)
    pd.DataFrame(df_result).to_csv(os.path.join(args.out, args.out_csv))
    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames],
                                fps=config["output_fps"])
    vid.write_videofile(os.path.join(args.out, args.out_video))

    # tmp_frame_dir = osp.dirname(frame_paths[0])
    # shutil.rmtree(tmp_frame_dir)


if __name__ == '__main__':
    main()