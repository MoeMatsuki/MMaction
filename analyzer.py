import copy as cp
import numpy as np
import cv2

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1

def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))

plate_blue = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
plate_blue = plate_blue.split('-')
plate_blue = [hex2color(h) for h in plate_blue]
plate_green = '004b23-006400-007200-008000-38b000-70e000'
plate_green = plate_green.split('-')
plate_green = [hex2color(h) for h in plate_green]

class Analyzer:
    def __init__(self, labels):
        self.labels = labels
        self.output_csv = "test.csv"

    def __call__(self, cfg, frames_path, frames, annotations):
        self.pre_processing(annotations, frames)
        vis_frames = self.visualize(frames_path, frames, annotations)
        vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames],
                                    fps=cfg.output_fps)
        csv_result = self.save_csv(frames_path, frames, annotations)
        return csv_result, vid

    def pre_processing(self, annotations, frames):
        self.nf, self.na = len(frames), len(annotations)
        print(self.nf, self.na)
        assert self.nf % self.na == 0
        self.nfpa = len(frames) // len(annotations)
        h, w, _ = frames[0].shape
        self.scale_ratio = np.array([w, h, w, h])

    def abbrev(self, name):
        """Get the abbreviation of label name:

        'take (an object) from (a person)' -> 'take ... from ...'
        """
        while name.find('(') != -1:
            st, ed = name.find('('), name.find(')')
            name = name[:st] + '...' + name[ed + 1:]
        return name

    def save_csv(self, frames_path, frames, annotations, plate=plate_blue):
        """Visualize frames with predicted annotations.

        Args:
            frames (list[np.ndarray]): Frames for visualization, note that
                len(frames) % len(annotations) should be 0.
            annotations (list[list[tuple]]): The predicted results.
            plate (str): The plate used for visualization. Default: plate_blue.
            max_num (int): Max number of labels to visualize for a person box.
                Default: 5.

        Returns:
            list[np.ndarray]: Visualized frames.
        """
        if plate == None:
            plate = self.plate_blue
        else:
            plate = plate

        anno = None
        frame_result = {
                    "frame": [],
                    "min_x": [],
                    "min_y": [],
                    "max_x": [],
                    "max_y": [],
                }
        [frame_result.update({l: []}) for l in self.labels]
        for i in range(self.na):
            anno = annotations[i]
            if anno is None:
                continue
            for j in range(self.nfpa):
                ind = i * self.nfpa + j
                for ann in anno:
                    box = ann[0]
                    label = ann[1]
                    if not len(label):
                        continue
                    score = ann[2]
                    box = (box * self.scale_ratio).astype(np.int64)
                    frame_result["frame"].append(frames_path[ind])
                    frame_result["min_x"].append(box[0])
                    frame_result["min_y"].append(box[1])
                    frame_result["max_x"].append(box[2])
                    frame_result["max_y"].append(box[3])
                    for lb in self.labels:
                        if lb in label:
                            k = label.index(lb)
                            frame_result[lb].append(str(score[k]))
                        else:
                            frame_result[lb].append(str(0))

        return frame_result

    def visualize(self, frames_path, frames, annotations, plate=plate_blue, max_num=5):
        """Visualize frames with predicted annotations.

        Args:
            frames (list[np.ndarray]): Frames for visualization, note that
                len(frames) % len(annotations) should be 0.
            annotations (list[list[tuple]]): The predicted results.
            plate (str): The plate used for visualization. Default: plate_blue.
            max_num (int): Max number of labels to visualize for a person box.
                Default: 5.

        Returns:
            list[np.ndarray]: Visualized frames.
        """
        assert max_num + 1 <= len(plate)
        plate = [x[::-1] for x in plate]
        frames_ = cp.deepcopy(frames)
        anno = None
        for i in range(self.na):
            anno = annotations[i]
            if anno is None:
                continue
            for j in range(self.nfpa):
                ind = i * self.nfpa + j
                frame = frames_[ind]
                for ann in anno:
                    box = ann[0]
                    label = ann[1]
                    if not len(label):
                        continue
                    score = ann[2]
                    box = (box * self.scale_ratio).astype(np.int64)
                    st, ed = tuple(box[:2]), tuple(box[2:])
                    cv2.rectangle(frame, st, ed, plate[0], 2)
                    for k, lb in enumerate(label):
                        if k >= max_num:
                            break
                        text = self.abbrev(lb)
                        text = ': '.join([text, str(score[k])])
                        location = (0 + st[0], 18 + k * 18 + st[1])
                        textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE,
                                                THICKNESS)[0]
                        textwidth = textsize[0]
                        diag0 = (location[0] + textwidth, location[1] - 14)
                        diag1 = (location[0], location[1] + 2)
                        cv2.rectangle(frame, diag0, diag1, plate[k + 1], -1)
                        cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                    FONTCOLOR, THICKNESS, LINETYPE)

        return frames_