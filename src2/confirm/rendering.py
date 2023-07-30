import cv2
import copy as cp
import numpy as np

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


def abbrev(name):
    """Get the abbreviation of label name:
    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name

def visualize(frames, annotations, plate=plate_blue, max_num=5):
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
                if not len(label):
                    continue
                if len(ann) == 3:
                    score = ann[2]
                box = (box * scale_ratio).astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                cv2.rectangle(frame, st, ed, plate[0], 2)
                for k, lb in enumerate(label):
                    if k >= max_num:
                        break
                    if len(ann) == 3:
                        text = abbrev(lb)
                        text = ': '.join([text, str(score[k])])
                    text = lb
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

def rendering_text(img, box, label, n=0, cols=plate_blue,):
    st, ed = tuple(box[:2]), tuple(box[2:])
    location = (0 + st[0], 18 + n * 18 + st[1])
    textsize = cv2.getTextSize(label, FONTFACE, FONTSCALE,
                                THICKNESS)[0]
    textwidth = textsize[0]
    diag0 = (location[0] + textwidth, location[1] - 14)
    diag1 = (location[0], location[1] + 2)
    cv2.rectangle(img, diag0, diag1, cols[n + 1], -1)
    cv2.putText(img, label, location, FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)
    return img

def rendering_bboxes(img, boxes, labels, cols=plate_blue, f_scaring=True):
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
        img_ = rendering_text(img_, box, labels[ix])
    return img_

def scaring(img, box):
    h, w, _ = img.shape[:3]
    scale_ratio = np.array([w, h, w, h])
    return (box * scale_ratio).astype(np.int64)