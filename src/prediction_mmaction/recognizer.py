import setup as setup
import mmcv
import numpy as np
import torch
import cv2
import os
from mmengine.runner import load_checkpoint
from mmaction.registry import MODELS

# from mmaction.models import build_detector

class Recognizer:
    def __init__(self, conf):
        self.device = conf.device
        self.action_score_thr = conf.action_score_thr
        # self.config  = formater_config(conf.config)
        self.config  = conf

        try:
            # In our spatiotemporal detection demo, different actions should have
            # the same number of bboxes.
            self.config['model']['test_cfg']['rcnn']['action_thr'] = .0
        except KeyError:
            pass

    def __call__(self, human_detections, timestamps, clip_len, frame_interval, frames, new_h, new_w):
        model = self.build_model()
        return self.prediction(model, human_detections, timestamps, clip_len, frame_interval, frames, new_h, new_w)

    def load_label_map(self, file_path):
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
        return {int(x[0]): x[1] for x in lines}

    # Load label_map
    def get_label(self):
        label_map = self.load_label_map(self.config["label_map"])
        try:
            if self.config['data']['train']['custom_classes'] is not None:
                label_map = {
                    id + 1: label_map[cls]
                    for id, cls in enumerate(self.config['data']['train']
                                            ['custom_classes'])
                }
        except KeyError:
            pass
        return label_map

    def build_model(self):
        self.config.model.backbone.pretrained = None
        model = MODELS.build(self.config.model)
        # model = build_detector(self.config.model, test_cfg=self.config.get('test_cfg'))

        load_checkpoint(model, self.config["checkpoint"], map_location='cpu')
        model.to(self.device)
        model.eval()
        return model

    def pack_result(self, human_detection, result, img_h, img_w):
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

    def get_norm(self):
        # Get img_norm_cfg
        img_norm_cfg = self.config.img_norm_cfg
        if 'to_rgb' not in img_norm_cfg and 'to_bgr' in img_norm_cfg:
            to_bgr = img_norm_cfg.pop('to_bgr')
            img_norm_cfg['to_rgb'] = to_bgr
        img_norm_cfg['mean'] = np.array(img_norm_cfg['mean'])
        img_norm_cfg['std'] = np.array(img_norm_cfg['std'])
        return img_norm_cfg

    def prediction(self, model, human_detections, timestamps, clip_len, frame_interval, frames, new_h, new_w):
        img_norm_cfg = self.get_norm()

        predictions = []

        label_map = self.get_label()

        print('Performing SpatioTemporal Action Detection for each clip')
        assert len(timestamps) == len(human_detections)
        prog_bar = mmcv.ProgressBar(len(timestamps))
        for timestamp, proposal in zip(timestamps, human_detections):
            if proposal.shape[0] == 0:
                predictions.append(None)
                continue

            start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
            window_size = clip_len * frame_interval
            frame_inds = start_frame + np.arange(0, window_size, frame_interval)
            frame_inds = list(frame_inds - 1)
            imgs = [frames[ind].astype(np.float32) for ind in frame_inds]
            _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in imgs]
            # THWC -> CTHW -> 1CTHW
            input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
            input_tensor = torch.from_numpy(input_array).to(self.device)
            with torch.no_grad():
                result = model(
                    return_loss=False,
                    img=[input_tensor],
                    img_metas=[[dict(img_shape=(new_h, new_w))]],
                    proposals=[[proposal]])
                result = result[0]
                prediction = []
                # N proposals
                for i in range(proposal.shape[0]):
                    prediction.append([])
                # Perform action score thr
                for i in range(len(result)):
                    if i + 1 not in label_map:
                        continue
                    for j in range(proposal.shape[0]):
                        try:
                            if result[i][j, 4] > self.action_score_thr:
                                prediction[j].append((label_map[i + 1], result[i][j,
                                                                                    4]))
                        except IndexError as e:
                            continue
                predictions.append(prediction)
            prog_bar.update()

        results = []
        for human_detection, prediction in zip(human_detections, predictions):
            results.append(self.pack_result(human_detection, prediction, new_h, new_w))
        return results