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
    if ': ' in lines[0]:
        lines = [x.strip().split(': ') for x in lines]
        return {int(x[0]): x[1] for x in lines}
    else:
        return {ix: x.split("\n")[0] for ix, x in enumerate(lines)}

def dense_timestamps(timestamps, n):
    """Make it nx frames."""
    old_frame_interval = (timestamps[1] - timestamps[0])
    start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
    new_frame_inds = np.arange(
        len(timestamps) * n) * old_frame_interval / n + start
    return new_frame_inds.astype(np.int64)

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
            frame_res = [(bboxes, label, score_)]
        pre_f_name = f_name
    result.append(frame_res)

    return result
        