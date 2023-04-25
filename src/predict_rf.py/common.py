import numpy as np
import cv2
import copy as cp

def load_label_map(file_path):
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
def get_label(config):
    label_map = load_label_map(config["label_map"])
    try:
        if config['data']['train']['custom_classes'] is not None:
            label_map = {
                # id + 1: label_map[cls]
                cls: label_map[cls]
                for id, cls in enumerate(config['data']['train']
                                        ['custom_classes'])
            }
    except KeyError:
        pass
    return label_map

def dense_timestamps(timestamps, n):
    """Make it nx frames."""
    old_frame_interval = (timestamps[1] - timestamps[0])
    start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
    new_frame_inds = np.arange(
        len(timestamps) * n) * old_frame_interval / n + start
    return new_frame_inds.astype(np.int64)

def abbrev(name):
    """Get the abbreviation of label name:
    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name

def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))

plate_blue = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
plate_blue = plate_blue.split('-')
plate_blue = [hex2color(h) for h in plate_blue]
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

    FONTFACE = cv2.FONT_HERSHEY_DUPLEX
    FONTSCALE = 0.5
    FONTCOLOR = (255, 255, 255)  # BGR, white
    MSGCOLOR = (128, 128, 128)  # BGR, gray
    THICKNESS = 1
    LINETYPE = 1


    assert max_num + 1 <= len(plate)
    plate = [x[::-1] for x in plate]
    frames_ = cp.deepcopy(frames)
    nf, na = len(frames), len(annotations)
    assert nf % na == 0
    nfpa = len(frames) // len(annotations)
    anno = None
    h, w, _ = frames[0].shape
    scale_ratio = np.array([w, h, w, h])
    print(scale_ratio)
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
                # box = (box * scale_ratio).astype(np.int64)
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

def convert_result(out_df, labels):
    action_score_thr = 0.5
    bbox = ["min_x","min_y","max_x","max_y"]
    result = []
    frame_res = []
    pre_f_name = None
    for ix in range(len(out_df)):
        f_name = out_df.loc[ix, ["frame"]].values.tolist()[0]
        scores = out_df.loc[ix, labels].values.tolist()
        index_score = [ix for ix, b in enumerate(scores) if b > action_score_thr]
        score_ = [scores[i] for i in index_score]
        label = [labels[i] for i in index_score]
        score_.insert(0, 1)
        label.insert(0, out_df.loc[ix, ["action_label"]].values.tolist()[0])
        bboxes = out_df.loc[ix, bbox].values.tolist()
        if pre_f_name is None or f_name == pre_f_name:
            frame_res.append((bboxes, label, score_))
        else:
            result.append(frame_res)
            frame_res = []
        pre_f_name = f_name
    result.append(frame_res)

    return result
        