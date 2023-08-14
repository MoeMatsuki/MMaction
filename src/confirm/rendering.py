import cv2
import copy as cp
import numpy as np
from natsort import natsorted

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5 #0.7
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
label_path = "/home/moe/MMaction/download/KOKUYO_data/annotations/classes_en3.txt"


def abbrev(name):
    """Get the abbreviation of label name:
    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name

def visualize(frames, annotations, label_map, plate=plate_blue, max_num=5):
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
    nf, na = len(frames), len(annotations)
    print(nf, na)
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
            frame = frames_[ind]
            for ann in anno:
                box = ann[0]
                label = ann[1]
                # if not len(label):
                #     continue
                if len(ann) == 3:
                    score = ann[2]
                box = (box * scale_ratio).astype(np.int64)
                frame = rendering_bbox(frame, box)
                frame = rendering_text(frame, box, label, label_map, top_n=max_num, scores=score)
    return frames_

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

####
def rendering_bbox(img, box, col=plate_blue[0]):
    """
    Args:
        img(np.array)
        box(np.array)
    """
    st, ed = tuple(box[:2]), tuple(box[2:])
    cv2.rectangle(img, st, ed, col, 2)
    return img

def rendering_text(img, box, labels, label_map, top_n=1, scores=None, cols=plate_blue):
    # label_map = load_label_map(label_path)
    labels = natsorted(labels)
    labels = [label_map[i] for i in labels]
    for n, label in enumerate(labels[:top_n]):
        st, ed = tuple(box[:2]), tuple(box[2:])
        location = (0 + st[0], 18 + n * 18 + st[1])
        if scores is not None:
            text = abbrev(label)
            label = ': '.join([text, str(scores[n])])
        textsize = cv2.getTextSize(label, FONTFACE, FONTSCALE,
                                    THICKNESS)[0]
        textwidth = textsize[0]
        diag0 = (location[0] + textwidth, location[1] - 14)
        diag1 = (location[0], location[1] + 2)
        cv2.rectangle(img, diag0, diag1, cols[n + 1], -1)
        cv2.putText(img, label, location, FONTFACE, FONTSCALE,
                        FONTCOLOR, THICKNESS, LINETYPE)
    return img

def rendering_bboxes(img, boxes, labels, label_top_n=1, cols=plate_blue, f_scaring=True):
    """
    Args:
        img(np.array)
        box(np.array)
    """
    img_ = cp.deepcopy(img)
    num = len(plate_blue)
    for ix, box in enumerate(boxes):
        if f_scaring:
            # bbox
            box = scaring(img_, box)
        ixx = ix % num
        img_ = rendering_bbox(img_, box, plate_blue[ixx])
        label_map = load_label_map(label_path)
        img_ = rendering_text(img_, box, labels[ix], label_map, top_n=label_top_n)
    return img_

def scaring(img, box):
    h, w, _ = img.shape[:3]
    scale_ratio = np.array([w, h, w, h])
    return (box * scale_ratio).astype(np.int64)

