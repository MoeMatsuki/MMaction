import setup
from utils import formater_config
import mmcv
import numpy as np
import torch
import cv2

from mmaction.models import build_detector

class Recognizer:
    def __init__(self, conf, preprocessing):
        self.device = conf.device
        self.action_score_thr = conf.action_score_thr
        self.config  = mmcv.Config.fromfile(conf.config)
        self.timestamps = preprocessing.timestamps
        self.clip_len = preprocessing.clip_len
        self.frame_interval = preprocessing.frame_interval
        self.frames = preprocessing.frames
        self.frame_paths = preprocessing.frame_paths
        self.img_norm_cfg = preprocessing.img_norm_cfg
        self.new_h = preprocessing.new_h
        self.new_w = preprocessing.new_w
        self.label_map = preprocessing.get_label()

        try:
            # In our spatiotemporal detection demo, different actions should have
            # the same number of bboxes.
            self.config.model['test_cfg']['rcnn']['action_thr'] = .0
        except KeyError:
            pass

    def __call__(self, human_detections):
        model = self.build_model()
        return self.prediction(model, human_detections)

    def build_model(self):
        self.config.model["backbone"]["pretrained"] = None
        model = build_detector(self.config.model, test_cfg=self.config.get('test_cfg'))

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

    def prediction(self, model, human_detections):
        predictions = []

        print('Performing SpatioTemporal Action Detection for each clip')
        print(len(self.timestamps), len(human_detections))
        assert len(self.timestamps) == len(human_detections)
        prog_bar = mmcv.ProgressBar(len(self.timestamps))
        for timestamp, proposal in zip(self.timestamps, human_detections):
            if proposal.shape[0] == 0:
                predictions.append(None)
                continue

            start_frame = timestamp - (self.clip_len // 2 - 1) * self.frame_interval
            window_size = self.clip_len * self.frame_interval
            frame_inds = start_frame + np.arange(0, window_size, self.frame_interval)
            frame_inds = list(frame_inds - 1)
            imgs = [self.frames[ind].astype(np.float32) for ind in frame_inds]
            _ = [mmcv.imnormalize_(img, **self.img_norm_cfg) for img in imgs]
            # THWC -> CTHW -> 1CTHW
            input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
            input_tensor = torch.from_numpy(input_array).to(self.device)

            with torch.no_grad():
                result = model(
                    return_loss=False,
                    img=[input_tensor],
                    img_metas=[[dict(img_shape=(self.new_h, self.new_w))]],
                    proposals=[[proposal]])
                result = result[0]
                prediction = []
                # N proposals
                for i in range(proposal.shape[0]):
                    prediction.append([])
                # Perform action score thr
                for i in range(len(result)):
                    if i + 1 not in self.label_map:
                        continue
                    for j in range(proposal.shape[0]):
                        try:
                            if result[i][j, 4] > self.action_score_thr:
                                prediction[j].append((self.label_map[i + 1], result[i][j,
                                                                                    4]))
                        except IndexError as e:
                            print(e)
                            continue
                predictions.append(prediction)
            prog_bar.update()

        results = []
        for human_detection, prediction in zip(human_detections, predictions):
            results.append(self.pack_result(human_detection, prediction, self.new_h, self.new_w))
        return results