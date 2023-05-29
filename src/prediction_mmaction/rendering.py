import cv2
import copy as cp
import numpy as np

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


def abbrev(name):
    """Get the abbreviation of label name:
    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name

def visualize_bbox(frames, bboxes, plate=plate_blue, max_num=5):
    """Visualize frames with predicted annotations.
    Args:
        frames (list[np.ndarray]): Frames for visualization, note that
            len(frames) % len(annotations) should be 0.
        bboxes (list[list[tuple]]): The predicted results.
        plate (str): The plate used for visualization. Default: plate_blue.
        max_num (int): Max number of labels to visualize for a person box.
            Default: 5.
    Returns:
        list[np.ndarray]: Visualized frames.
    """

    assert max_num + 1 <= len(plate)
    plate = [x[::-1] for x in plate]
    frames_ = cp.deepcopy(frames)
    nf, na = len(frames), len(bboxes)
    assert nf % na == 0
    nfpa = len(frames) // len(bboxes)
    h, w, _ = frames[0].shape
    for i in range(na):
        bbox = bboxes[i]

        for j in range(nfpa):
            ind = i * nfpa + j
            frame = frames_[ind]
            for box in bbox:
                box = box.to('cpu').detach().numpy().copy().astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                cv2.rectangle(frame, st, ed, plate[0], 2)

    return frames_

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
                score = ann[2]
                box = (box * scale_ratio).astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                cv2.rectangle(frame, st, ed, plate[0], 2)
                for k, lb in enumerate(label):
                    if k >= max_num:
                        break
                    text = abbrev(lb)
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