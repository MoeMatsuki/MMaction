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
import mmengine
import mmcv
import numpy as np
import torch
from natsort import natsorted
from mmengine.config import DictAction
from mmaction.apis import detection_inference
from mmaction.structures import ActionDataSample
from mmengine.structures import InstanceData
from mmaction.registry import MODELS
from mmengine.runner import load_checkpoint

# from mmcv.runner import load_checkpoint

# from mmaction.models import build_detector

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

def load_label_map(file_path):
    """Load Label Map.

    Args:
        file_path (str): The file path of label map.
    Returns:
        dict: The label map (int -> label name).
    """
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}

def pack_result(human_detection, result, img_h, img_w):
    """Short summary.

    Args:
        human_detection (np.ndarray): Human detection result.
        result (type): The predicted label of each human proposal.
        img_h (int): The image height.
        img_w (int): The image width.
    Returns:
        tuple: Tuple of human proposal, label name and label score.
    """
    human_detection[:, 0::2] /= img_w
    human_detection[:, 1::2] /= img_h
    results = []
    if result is None:
        return None
    for prop, res in zip(human_detection, result):
        res.sort(key=lambda x: -x[1])
        results.append(
            (prop.data.cpu().numpy(), [x[0] for x in res], [x[1]
                                                            for x in res]))
    return results

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
    # config = mmcv.Config.fromfile(args.conf_path)
    config = mmengine.Config.fromfile(args.conf_path)
    config.merge_from_dict(config["cfg_options"])
    val_pipeline = config.val_pipeline

    sampler = [x for x in val_pipeline if x['type'] == 'SampleAVAFrames'][0]
    clip_len, frame_interval = sampler['clip_len'], sampler['frame_interval']
    window_size = clip_len * frame_interval
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'
    # Note that it's 1 based here
    timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
                           config["predict_stepsize"])

    label_map = load_label_map(config.label_map)
    try:
        if config['data']['train']['custom_classes'] is not None:
            label_map = {
                id + 1: label_map[cls]
                for id, cls in enumerate(config['data']['train']
                                         ['custom_classes'])
            }
    except KeyError:
        pass

    # Get Human detection results
    # detector = Detector(config)
    # human_detections, all_detection = detector(frame_paths, timestamps, w_ratio, h_ratio)
    center_frames = [frame_paths[ind - 1] for ind in timestamps]

    human_detections, _ = detection_inference(config.det_config,
                                              config.det_checkpoint,
                                              center_frames,
                                              config.det_score_thr,
                                              config.det_cat_id, config.device)
    # detector.save_results(all_detection, frames)
    torch.cuda.empty_cache()
    for i in range(len(human_detections)):
        det = human_detections[i]
        det[:, 0:4:2] *= w_ratio
        det[:, 1:4:2] *= h_ratio
        human_detections[i] = torch.from_numpy(det[:, :4]).to(config.device)

    # Get img_norm_cfg
    img_norm_cfg = dict(
        mean=np.array(config.model.data_preprocessor.mean),
        std=np.array(config.model.data_preprocessor.std),
        to_rgb=False)
    # img_norm_cfg = config['img_norm_cfg']
    # if 'to_rgb' not in img_norm_cfg and 'to_bgr' in img_norm_cfg:
    #     to_bgr = img_norm_cfg.pop('to_bgr')
    #     img_norm_cfg['to_rgb'] = to_bgr
    # img_norm_cfg['mean'] = np.array(img_norm_cfg['mean'])
    # img_norm_cfg['std'] = np.array(img_norm_cfg['std'])

# Build STDET model
    try:
        # In our spatiotemporal detection demo, different actions should have
        # the same number of bboxes.
        config['model']['test_cfg']['rcnn'] = dict(action_thr=0)
    except KeyError:
        pass


    config.model.backbone.pretrained = None
    model = MODELS.build(config.model)

    load_checkpoint(model, config.checkpoint, map_location='cpu')
    model.to(config.device)
    model.eval()

    predictions = []

    img_norm_cfg = dict(
        mean=np.array(config.model.data_preprocessor.mean),
        std=np.array(config.model.data_preprocessor.std),
        to_rgb=False)

    print('Performing SpatioTemporal Action Detection for each clip')
    assert len(timestamps) == len(human_detections)
    prog_bar = mmengine.ProgressBar(len(timestamps))
    for timestamp, proposal in zip(timestamps, human_detections):
        if proposal.shape[0] == 0:
            predictions.append(None)
            continue

        start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        frame_inds = start_frame + np.arange(0, window_size, frame_interval)
        frame_inds = list(frame_inds - 1)
        imgs = [frames[ind].astype(np.float32) for ind in frame_inds]
        _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in imgs]
        # THWC -> CTHW -> 1CTHW
        input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
        input_tensor = torch.from_numpy(input_array).to(config.device)

        datasample = ActionDataSample()
        datasample.proposals = InstanceData(bboxes=proposal)
        datasample.set_metainfo(dict(img_shape=(new_h, new_w)))
        with torch.no_grad():
            result = model(input_tensor, [datasample], mode='predict')
            scores = result[0].pred_instances.scores
            prediction = []
            # N proposals
            for i in range(proposal.shape[0]):
                prediction.append([])
            # Perform action score thr
            for i in range(scores.shape[1]):
                if i not in label_map:
                    continue
                for j in range(proposal.shape[0]):
                    if scores[j, i] > config.action_score_thr:
                        prediction[j].append((label_map[i], scores[j,
                                                                   i].item()))
            predictions.append(prediction)
        prog_bar.update()

    results = []
    for human_detection, prediction in zip(human_detections, predictions):
        results.append(pack_result(human_detection, prediction, new_h, new_w))

    def dense_timestamps(timestamps, n):
        """Make it nx frames."""
        old_frame_interval = (timestamps[1] - timestamps[0])
        start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
        new_frame_inds = np.arange(
            len(timestamps) * n) * old_frame_interval / n + start
        return new_frame_inds.astype(np.int64)

    dense_n = int(config.predict_stepsize / config.output_stepsize)
    frames = [
        cv2.imread(frame_paths[i - 1])
        for i in dense_timestamps(timestamps, dense_n)
    ]
    print('Performing visualization')
    vis_frames = visualize(frames, results)
    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames],
                                fps=config.output_fps)
    vid.write_videofile(args.out_video)

    # tmp_dir.cleanup()


if __name__ == '__main__':
    main()