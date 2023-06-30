import pandas as pd
import numpy as np
import argparse
import mmengine
from sklearn.metrics import classification_report

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('conf_path')
    args = parser.parse_args()
    return args

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


def convert(csv_file):
    frame_res = {}
    df = pd.read_csv(csv_file)
    f_split = lambda x: "_".join(x.split("_")[:-2]) + "_" + x.split("_")[-1]
    df["person_id"] = df["frame"].map(f_split)
    f_split = lambda x: x.split("_")[-2]
    df["f"] = df["frame"].map(f_split)
    person_unique = df["person_id"].unique()
    for u in person_unique:
        frame_res.update({u:{}})
        df_u = df[df["person_id"] == u]
        df_u = df_u.sort_values('f')

        res = dict(df_u.loc[:, label_name])
        # num = len(df_u)
        # count = 0
        # while count < num:
        #     if count+time_w < num:
        #         df_u_time = df_u[count:count+time_w]
        #     else:
        #         df_u_time = df_u[count:num]
        #     count = count + time_w
        #     res = dict(df_u_time.loc[:, label_name].mean())
        #     for i,k in res.items():
        #         if k > 0.5:
        #             res[i] = 1
        #         else:
        #             res[i] = 0
        #     frame_key = str(count)+"-"+str(count+time_w)
        #     frame_res[u].update({frame_key: res})
    return frame_res


# args = parse_args()
# config = mmengine.Config.fromfile(args.conf_path)
# config.merge_from_dict(config["cfg_options"])
# import cfg

g_csv = "/home/moe/MMaction/fastlabel1/annotations/train_rf.csv"
pre_csv = "/home/moe/MMaction/test_mmaction.csv"
time_w = 10
label = "/home/moe/MMaction/download/KOKUYO_data/annotations/classes_en.txt"

label_name = list(load_label_map(label).values())
g_convert = convert(g_csv)
print(g_convert)
# pre_convert = convert(pre_csv)
# g_list = []
# pre_list = []
# for img, f_list in pre_convert.items():
#     for f,a in f_list.items():
#         g_list.append(list(g_convert[img][f].values()))
#         pre_list.append(list(a.values()))
# print(classification_report(np.array(g_list), np.array(pre_list)))