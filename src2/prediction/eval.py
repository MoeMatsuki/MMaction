import pandas as pd
import math
import cv2
import os
import argparse
import numpy as np
from copy import copy
from sklearn.metrics import classification_report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default="result")
    parser.add_argument('--outdir', default=None)
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

def convert_frame_name1(frame_str):
    frame_num = frame_str.split(".")[0]
    frame_num = int(int(frame_num.split("_")[-1]))#)/30)
    # frame_num = str(frame_num).zfill(6)
    # img_base = frame_str.split("/")[-2]
    # return img_base + "_" + frame_num
    return frame_num

def convert_frame_name2(frame_str):
    return int(frame_str.split("_")[-2])

def convert_frame_name3(frame_str):
    return frame_str.split("_")[-1]

def convert_frame_name4(frame_str):
    return "_".join(frame_str.split("_")[:-2])

def convert_frame_name5(frame_str):
    return "_".join(frame_str.split("_")[:-1])

def get_size(img_path):
    img = cv2.imread(img_path)
    h, w, _  = img.shape[:3]
    return h, w

def preprocess_pre(df):
    
    df["frame_num"] = df["frame"].map(convert_frame_name1)
    df["min_point"] = df["min_x"]**2 + df["min_y"]**2
    df["img"] = df["frame"].map(convert_frame_name5)
    df["min_point"] = df["min_point"].map(math.sqrt)
    # h,w = get_size(df_pre["frame"].values.tolist()[0])
    return df

def preprocess_gt(df):
    df["frame_num"] = df["frame"].map(convert_frame_name2)
    df["person"] = df["frame"].map(convert_frame_name3)
    df["img"] = df["frame"].map(convert_frame_name4)
    df["min_point"] = df["min_x"]**2 + df["min_y"]**2
    df["min_point"] = df["min_point"].map(math.sqrt)
    return df

def process(out, label_name):
    # 定数
    pre_csv = os.path.join(out, "test_mmaction.csv")
    out_csv = os.path.join(out, "low_activity.csv")
    # train_csv = "/home/moe/MMaction/fastlabel1/annotations/train_rf.csv"
    train_csv = "/home/moe/MMaction/fastlabel1/annotations/val_rf.csv"
    label_name_append = copy(label_name)
    label_name_append.append("min_point")
    img = out.split("/")[-1]


    # MMactionの結果を読み込み
    try:
        df_pre = pd.read_csv(pre_csv)
    except FileNotFoundError:
        return None

    # 予測結果のデータ整理
    df_pre = preprocess_pre(df_pre)

    # 正解結果のデータ整理
    df_gt = pd.read_csv(train_csv)
    df_gt = preprocess_gt(df_gt)

    # フィルタリング
    df_gt = df_gt[df_gt["img"] == img]
    print(df_gt)

    pre_frame = 0
    pre_res = []
    gt_res = []
    all_result = []
    ave_result = []
    for frame_n, df_pre_f in df_pre.groupby("frame_num"):
        # 予測結果をフレーム画像ごとにソート
        df_pre_f = df_pre_f.sort_values('min_point')
        bboxes = df_pre_f.loc[:, ["min_x", "min_y", "max_x", "max_y"]].values
        
        # 正解のデータのフレームを探索
        try:
            df_gt_ = df_gt[df_gt["frame_num"] >= pre_frame]
            df_gt_ = df_gt[df_gt["frame_num"] <= frame_n]
        except ValueError:
            return None
        df_gt_ = df_gt_.sort_values('min_point')

        gt_list = []
        for p, df_gt_p in df_gt_.groupby("person"):
            df_gt_p = df_gt_p.loc[:, label_name_append].mean()
            res = list(df_gt_p)[:-1]
            for i in range(len(res)):
                if res[i] > 0.5:
                    res[i] = 1
                else:
                    res[i] = 0
            gt_list.append(res)
        pre_frame = frame_n + 1

        df_pre_vec = df_pre_f.loc[:, label_name]
        # df_pre_vec = df_pre_vec.mask(df_pre_vec>=0.5,1)
        # print(df_pre_vec)
        pre = df_pre_vec.T.to_dict()
        pre_list = []
        for i,k in pre.items():
            for p,kk in k.items():
                if kk > 0.5:
                    k[p] = 1
                else:
                    k[p] = 0
            pre_list.append(list(k.values()))
        # for ix, _ in enumerate(pre_list):
        #     report = classification_report(np.array([gt_list[ix]]), np.array([pre_list[ix]]), target_names=label_name, output_dict=True)
        #     all_result.append({frame_n: {ix: {"report":report, "bbox": bboxes[ix]}}})
        if len(gt_list) != len(pre_list):
            if len(gt_list) > len(pre_list):
               diff = len(gt_list) - len(pre_list)
               for i in range(diff):
                    pre_list.append([0] * len(label_name))
            if len(gt_list) < len(pre_list):
               diff = len(pre_list) - len(gt_list)
               for i in range(diff):
                    gt_list.append([0] * len(label_name))
        ave_report = classification_report(np.array(gt_list), np.array(pre_list), target_names=label_name, output_dict=True)
        pre_support = pd.Series(np.array(pre_list).sum(axis=0))
        for ix, l in enumerate(label_name):
            ave_report[l].update({"pre_count": pre_support[ix]})
        ave_result.append(pd.DataFrame(ave_report).T)
        

    # ave_report = classification_report(np.array(gt_res), np.array(pre_res), target_names=label_name, output_dict=True)
    # pre_support = pd.Series(np.array(pre_res).sum(axis=0))
    # for ix, l in enumerate(label_name):
    #     ave_report[l].update({"pre_count": pre_support[ix]})
    for ix, dd in enumerate(ave_result):
        if ix == 0:
            df = dd
        else:
            df = df + dd
    report_df_ave = df.iloc[:, :3] / len(ave_result)
    report_df_ave = pd.concat([report_df_ave, df.iloc[:, 3:]], axis = 1)
    # report_df = pd.DataFrame(ave_report).T
    report_df_ave.to_csv(out_csv)
    return report_df_ave

def main():
    label = "/home/moe/MMaction/fastlabel1/annotations/classes_en.txt"
    label_name = list(load_label_map(label).values())
    args = parse_args()

    if args.outdir is None:
        res = process(args.out, label_name)
    else:
        res_list = []
        dirs = os.listdir(args.outdir)
        for d in dirs:
            d_path = os.path.join(args.outdir, d)
            if os.path.isdir(d_path):
                res = process(d_path, label_name)
                if res is not None:
                    res_list.append(res)
                    # res_list.append(list(res.values()))
        # print(np.mean(np.array(res_list), axis=0))
        for ix, d in enumerate(res_list):
            if ix == 0:
                all_d = d
            else:
                all_d = all_d + d
        ave_d = all_d/len(res_list)
        res = pd.concat([ave_d.iloc[:, :3], all_d.iloc[:, 3:]], axis=1)
        print(res)
        res.to_csv("ave_result.csv")

if __name__ == "__main__":
    main()
