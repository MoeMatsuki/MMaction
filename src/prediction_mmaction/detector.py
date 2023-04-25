import torch
import mmcv

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

    def __call__(self, frame_paths, timestamps, w_ratio, h_ratio):
        # Get Human detection results
        center_frames = [frame_paths[ind - 1] for ind in timestamps]
        human_detections = self.detection_inference(center_frames)
        for i in range(len(human_detections)):
            det = human_detections[i]
            det[:, 0:4:2] *= w_ratio
            det[:, 1:4:2] *= h_ratio
            human_detections[i] = torch.from_numpy(det[:, :4]).to(self.device)
        return human_detections

    def detection_inference(self, frame_paths):
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
        print('Performing Human Detection for each frame')
        prog_bar = mmcv.ProgressBar(len(frame_paths))
        for frame_path in frame_paths:
            result = inference_detector(model, frame_path)
            # We only keep human detections with score larger than det_score_thr
            result = result[0][result[0][:, 4] >= self.det_score_thr]
            results.append(result)
            prog_bar.update()
        return results