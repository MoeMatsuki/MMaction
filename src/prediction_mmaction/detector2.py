import torch
import mmcv
import cv2
import os
from rendering import visualize_bbox

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                    '`init_detector` form `mmdet.apis`. These apis are '
                    'required in this demo! ')  

class Detector:
    def __init__(self, conf):
        self.device = conf.device
        self.det_config = conf.det_config
        self.det_checkpoint = conf.det_checkpoint
        self.det_score_thr = conf.det_score_thr
        self.timestamps = []

    def __call__(self, frame_paths, timestamps, w_ratio, h_ratio):
        # Get Human detection results
        self.timestamps = timestamps
        self.frame_paths = [frame_paths[ind - 1] for ind in timestamps]
        human_detections,all_detection = self.detection_inference()
        for i in range(len(human_detections)):
            det = human_detections[i]
            det[:, 0:4:2] *= w_ratio
            det[:, 1:4:2] *= h_ratio
            human_detections[i] = torch.from_numpy(det[:, :4]).to(self.device)
        for i in range(len(all_detection)):
            for j in range(len(all_detection[i])):
                det = all_detection[i][j]
                det[:, 0:4:2] *= w_ratio
                det[:, 1:4:2] *= h_ratio
                all_detection[i][j] = torch.from_numpy(det[:, :4]).to(self.device)
        return human_detections, all_detection

    def detection_inference(self):
        """Detect human boxes given frame paths.

        Args:
            args (argparse.Namespace): The arguments.
            frame_paths (list[str]): The paths of frames to do detection inference.

        Returns:
            list[np.ndarray]: The human detection results.
        """
        model= init_detector(self.det_config, self.det_checkpoint, self.device)
        assert model.CLASSES[0] == 'person', ('We require you to use a detector '
                                            'trained on COCO')
        results = []
        all_results = []
        print('Performing Human Detection for each frame')
        prog_bar = mmcv.ProgressBar(len(self.frame_paths))
        for frame_path in self.frame_paths:
            result = inference_detector(model, frame_path)
            # We only keep human detections with score larger than det_score_thr
            result_human = result[0][result[0][:, 4] >= self.det_score_thr]
            results.append(result_human)
            res= []
            for i in range(len(result)):
                all_r = result[i][result[i][:, 4] >= self.det_score_thr]
                res.append(all_r)
            all_results.append(res)
            prog_bar.update()
        return results, all_results

    def save_results(self, results, frames):
        frames = [cv2.imread(os.path.join("tmp_resize", path)) for path in self.frame_paths]
        # frames = [frames[ind - 1] for ind in self.timestamps]
        re = []
        for f in results:
            re.append([bb for c in f for bb in c])
        vis_frames = visualize_bbox(frames, re)
        dirname = os.path.dirname(os.path.join("tmp_det", self.frame_paths[0]))
        os.makedirs(dirname, exist_ok=True)
        for i, f in enumerate(vis_frames):
            cv2.imwrite(os.path.join("tmp_det", self.frame_paths[i]), f)

