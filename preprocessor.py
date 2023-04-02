import os
import os.path as osp
import cv2
import mmcv
import numpy as np
from importlib import import_module 

from utils import DictDotNotation, formater_config

class Preprocessor:
    def __init__(self, conf):
        self.config  = formater_config(conf.config)
        self.cfg_options = DictDotNotation(conf.cfg_options)
        self.predict_stepsize = conf.predict_stepsize
        self.label_map = conf.label_map

        # result
        self.timestamps = None
        self.w_ratio = None
        self.h_ratio = None
        self.frames = None
        self.frame_paths = None
        self.clip_len = None
        self.frame_interval = None
        self.window_size = None
        self.new_w = None
        self.new_h = None
    def __call__(self, video):

        self.frame_paths, original_frames = self.frame_extraction(video)
        num_frame = len(self.frame_paths)
        h, w, _ = original_frames[0].shape

        # resize frames to shortside 256
        self.new_w, self.new_h = mmcv.rescale_size((w, h), (256, np.Inf))
        self.frames = [mmcv.imresize(img, (self.new_w, self.new_h)) for img in original_frames]
        self.w_ratio, self.h_ratio = self.new_w / w, self.new_h / h
        self.timestamps = self.get_clip(num_frame)
        self.get_norm()

    def frame_extraction(self, video_path):
        """Extract frames given video_path.

        Args:
            video_path (str): The video_path.
        """
        frames = []
        frame_paths = []
        # Load the video, extract frames into ./tmp/video_name
        target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
        if os.path.exists(target_dir):
            frame_paths = os.listdir(target_dir)
            frame_paths = [os.path.join(target_dir, f) for f in frame_paths]
            for f in frame_paths:
                frames.append(cv2.imread(f))
            return frame_paths, frames
        else:
            os.makedirs(target_dir, exist_ok=True)
            # Should be able to handle videos up to several hours
            frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
            vid = cv2.VideoCapture(video_path)
            flag, frame = vid.read()
            cnt = 0
            while flag:
                frames.append(frame)
                frame_path = frame_tmpl.format(cnt + 1)
                frame_paths.append(frame_path)
                cv2.imwrite(frame_path, frame)
                cnt += 1
                flag, frame = vid.read()
            return frame_paths, frames

    def get_clip(self, num_frame):
        # Get clip_len, frame_interval and calculate center index of each clip
        # config = mmcv.Config.fromfile(self.config)
        # self.config.merge_from_dict(self.cfg_options)
        for k, v in self.cfg_options.items():
            self.config[k] = v
        val_pipeline = self.config.data["val"]["pipeline"]

        sampler = [x for x in val_pipeline if x['type'] == 'SampleAVAFrames'][0]
        print(sampler)
        clip_len, frame_interval = sampler['clip_len'], sampler['frame_interval']
        window_size = clip_len * frame_interval
        assert clip_len % 2 == 0, 'We would like to have an even clip_len'
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        # Note that it's 1 based here
        timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
                            self.predict_stepsize)
        self.window_size = window_size
        return timestamps

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

    def get_label(self):
        # Load label_map
        label_map = self.load_label_map(self.label_map)
        print(self.config)
        try:
            if self.config.data['train']['custom_classes'] is not None:
                label_map = {
                    id + 1: label_map[cls]
                    for id, cls in enumerate(self.config.data['train']
                                            ['custom_classes'])
                }
        except KeyError:
            pass

        return label_map
    
    def get_norm(self):
        # Get img_norm_cfg
        img_norm_cfg = self.config.img_norm_cfg
        if 'to_rgb' not in img_norm_cfg and 'to_bgr' in img_norm_cfg:
            to_bgr = img_norm_cfg.pop('to_bgr')
            img_norm_cfg['to_rgb'] = to_bgr
        img_norm_cfg['mean'] = np.array(img_norm_cfg['mean'])
        img_norm_cfg['std'] = np.array(img_norm_cfg['std'])
        self.img_norm_cfg = img_norm_cfg