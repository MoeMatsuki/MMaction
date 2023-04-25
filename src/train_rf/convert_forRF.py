import pandas as pd
import mmcv

# df = load_csv("test_mmaction2.csv")
col_name = ["img_name", "frame_sec", "min_x", "min_y", "max_x", "max_y", "action_id", "person_id"]
df_gt = pd.read_csv("/home/moe/MMaction/KOKUYO_data/annotations/train.csv", header=None)
conf_path = "config/default.py"
config = mmcv.Config.fromfile(conf_path)

df_gt = df_gt.set_axis(col_name, axis='columns')
convert_str = lambda x: str(x).zfill(5)
df_gt["id"] = df_gt["img_name"] + "_" + df_gt["frame_sec"].map(convert_str) + "_" + df_gt["person_id"].map(convert_str)
print(df_gt)
group = df_gt.groupby("id")


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
def get_label():
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

action_label = load_label_map("/home/moe/MMaction/KOKUYO_data/annotations/classes_en2.txt")
action_label_ids = list(action_label.keys())
label_map = get_label()
print(label_map)
labels = list(label_map.values())
frame_result = {
    "frame":[],
    "min_x": [],
    "min_y": [],
    "max_x": [],
    "max_y": [],
    "action_label":["NaN"] * len(group.groups),
    "action_id":[0] * len(group.groups)
}
[frame_result.update({l: [0] * len(group.groups)}) for l in labels]
count = 0
for id, ix_list in group.groups.items():
    frame_result["frame"].append(id)
    l = df_gt.iloc[ix_list[0], :]
    frame_result["min_x"].append(l["min_x"])
    frame_result["min_y"].append(l["min_y"])
    frame_result["max_x"].append(l["max_x"])
    frame_result["max_y"].append(l["max_y"])
    for ix in ix_list:
        action_id = df_gt.iloc[ix, 6]
        if action_id in action_label_ids:
            frame_result["action_id"][count] = action_id
            frame_result["action_label"][count] = action_label[action_id]
            continue
        action_name = label_map[action_id]
        if action_name in labels:
            frame_result[action_name][count] = 1
    count += 1
for k,v in frame_result.items():
    print(k, len(v))
result = pd.DataFrame(frame_result)
print(result)
result.to_csv("test_rf.csv")