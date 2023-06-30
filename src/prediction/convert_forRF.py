"""convert from train.csv to test_rf.csv for imprement randomforest

アノテーション済みのtrain.csv からtest_rf.csv に変換
    train.csv: アノテーション済みのcsvファイル。
        columns: ファイル名, フレーム数, bbox座標(min_x,min_y,max_x,max_y),activity_id, person_id 
    train_rf.csv: ランダムフォレスト用に変換したcsvファイル。１行は１画像内の一人分。
        columns: ファイル名_フレーム数_personid, bbox座標, 含まれるactivityid, action_id(目的変数), 下位行動（説明変数） 

    flag_multi_cls=Trueにした場合、上位行動の他クラス分類となる
    flag_multi_cls=Falseにした場合、true_idsで指定したラベルを１、それ以外を０とする
        ※ exclude_sample_idsでした行動IDを持つサンプルは除外される。
        ※ 上位行動のラベルを含んでいないサンプルは除外される。

"""

import pandas as pd
import argparse
import csv
import copy
from common import load_label_map
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np
from statistics import mean

def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('--config', default=None, help='train config file path')
    args = parser.parse_args()
    return args

class Object:
    def __init__(self, bbox, act, low, id):
        self.bboxes = [bbox]
        self.acts = [act]
        self.low = [low]
        self.id = id
        self.high = -1
        self.frames_since_seen = 0
        self.low_posture = [0.0] * 4 # 10-
        self.low_body_dir = [0.0] * 2 # 20-
        self.low_upper_slope = [0.0] * 3 # 30-
        self.low_upper_move = [0.0] * 2 # 40-
        self.low_hand_pos = [0.0] * 10 # 50-
        self.low_hand_move = [0.0] * 10 # 60-
        self.low_face_dir = [0.0] * 3 # 70-
        self.low_face_slope = [0.0] * 3 # 80-
        self.low_face_move = [0.0] * 3 # 90-
        self.low_eye_target = [0.0] * 9 # 100-
        self.low_hand_hold = [0.0] * 10 # 110-
        self.low_motion = [0.0] * 10 # 120-
        self.low_other = [0.0] * 3 # 130-
        self.low_all = []
        self.avg_bbox = []

    def add_bbox(self, bbox):
        self.bboxes.append(bbox)
        self.frames_since_seen = 0  # reset the counter

    def add_low(self, low):
        self.low.append(low)

    def update_not_seen(self):
        self.frames_since_seen += 1  # increase the counter

class ConvertRF:
    def __init__(self, config):
        self.col_name = ["img_name", "frame_sec", "min_x", "min_y", "max_x", "max_y", "action_id", "person_id"]

        # csvファイル
        self.rf_train_csv = config.rf_train_csv
        self.rf_val_csv = config.rf_val_csv

        # 上位行動ラベル
        self.action_label_path = config.action_label
        # 下位行動ラベル
        self.label_map = config.label_map

        # 対象として扱う出現フレーム数の閾値
        self.frm_thresh = config.frm_thresh

        # 多クラス分類かどうか
        self.multi_cls = config.flag_multi_cls
        # ２値の場合のTrueとする行動IDのリスト
        self.true_ids = config.true_ids

        # このidを含むサンプルは除外される
        self.exclude_sample_ids = config.exclude_sample_ids
        self.set_label()


    def set_label(self):
        # 上位行動
        self.h_action_label = load_label_map(self.action_label_path)
        self.h_action_label_ids = list(self.h_action_label.keys())

        # 下位行動
        self.l_action_label = load_label_map(self.label_map)
        self.l_action_names = list(self.l_action_label.values())

        # 下位行動が目的変数の場合、説明変数から下位行動を削除する
        if len(self.true_ids) != 0:
            for true_id in self.true_ids:
                if true_id in self.h_action_label_ids:
                    continue
                del self.l_action_label[true_id]

    def __call__(self, input, output):
        train_df = pd.read_csv(input, header=None)
        train_df = self.processing(train_df)
        train_df.to_csv(output, index=False)

    def has_duplicates(self, seq):
        return len(seq) != len(set(seq))

    def compute_distance(self, bbox1, bbox2):
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    def traceObjects(self, prev_objects, curr_bboxes, curr_act):
        max_frames_not_seen = 10 # threshold for identification
        curr_object_ids = [-1]*len(curr_bboxes)  # initialize object ids for current frame

        # If there is no previous object, assign new ids
        if len(prev_objects) == 0:
            return list(range(len(curr_bboxes)))

        # Compute cost matrix
        cost_matrix = np.full((len(prev_objects), len(curr_bboxes)), fill_value=1000000000)
        for i, obj in enumerate(prev_objects):
            for j, (bbox, act) in enumerate(zip(curr_bboxes, curr_act)):
                if obj.acts[-1] == act:  # Only calculate distance if actions match and get the latest action
                    cost_matrix[i, j] = self.compute_distance(obj.bboxes[-1], bbox)  # get the latest bbox

        # Remove rows and columns with only np.inf
        row_mask = np.all(cost_matrix == 1000000000, axis=1)
        col_mask = np.all(cost_matrix == 1000000000, axis=0)
        cost_matrix = cost_matrix[~row_mask, :]
        cost_matrix = cost_matrix[:, ~col_mask]

        # If cost matrix is empty, assign new ids
        if cost_matrix.size == 0 or cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0:
            return list(range(len(curr_bboxes)))

        # If cost matrix is not empty, find optimal assignment
        row_inds, col_inds = linear_sum_assignment(cost_matrix)
        # Update object ids for the current frame
        for row_ind, col_ind in zip(row_inds, col_inds):
            if cost_matrix[row_ind, col_ind] < np.inf:
                # Convert column indices back to original indices
                original_col_ind = np.arange(len(curr_bboxes))[~col_mask][col_ind]
                original_row_inds = [i for i, mask in enumerate(row_mask) if not mask]
                curr_object_ids[original_col_ind] = prev_objects[original_row_inds[row_ind]].id

        # Assign new ids to unassigned bboxes
        max_id = max([obj.id for obj in prev_objects if obj.frames_since_seen <= max_frames_not_seen], default=-1) + 1
        for i in range(len(curr_object_ids)):
            if curr_object_ids[i] == -1:
                curr_object_ids[i] = max_id
                max_id += 1

        return curr_object_ids

    def update_objects(self, prev_objects, curr_bboxes, curr_act, curr_low):
        curr_object_ids = self.traceObjects(prev_objects, curr_bboxes, curr_act)
        new_objects = []
        # update seen objects and add unseen objects to new_objects
        for obj in prev_objects:
            if obj.id in curr_object_ids:
                ind = curr_object_ids.index(obj.id)
                obj.add_bbox(curr_bboxes[ind])
                obj.add_low(curr_low[ind])
            else:
                obj.update_not_seen()
            new_objects.append(obj)

        # add new objects to new_objects
        max_id = max([obj.id for obj in prev_objects], default=-1) + 1
        for i, curr_id in enumerate(curr_object_ids):
            if curr_id >= max_id:
                new_objects.append(Object(curr_bboxes[i], curr_act[i], curr_low[i], curr_id))

        return new_objects

    def get_average_bbox(self, objects): # 平均bboxを算出
        for obj in objects:
            bboxes = obj.bboxes
            min_x_values = [bbox[0] for bbox in bboxes]
            min_y_values = [bbox[1] for bbox in bboxes]
            max_x_values = [bbox[2] for bbox in bboxes]
            max_y_values = [bbox[3] for bbox in bboxes]

            avg_min_x = mean(min_x_values)
            avg_min_y = mean(min_y_values)
            avg_max_x = mean(max_x_values)
            avg_max_y = mean(max_y_values)

            obj.avg_bbox = [avg_min_x, avg_min_y, avg_max_x, avg_max_y]

    def count_low(self, objects): # 下位行動カウント
        for obj in objects:
            for line in obj.low:
                for i in range(len(line)):
                    j = line[i] % 10
                    if line[i] < 20:
                        obj.low_posture[j] += 1
                    elif line[i] < 30:
                        obj.low_body_dir[j] += 1
                    elif line[i] < 40:
                        obj.low_upper_slope[j] += 1
                    elif line[i] < 50:
                        obj.low_upper_move[j] += 1
                    elif line[i] < 60:
                        obj.low_hand_pos[j] += 1
                    elif line[i] < 70:
                        obj.low_hand_move[j] += 1
                    elif line[i] < 80:
                        obj.low_face_dir[j] += 1
                    elif line[i] < 90:
                        obj.low_face_slope[j] += 1
                    elif line[i] < 100:
                        obj.low_face_move[j] += 1
                    elif line[i] < 110:
                        obj.low_eye_target[j] += 1
                    elif line[i] < 120:
                        obj.low_hand_hold[j] += 1
                    elif line[i] < 130:
                        obj.low_motion[j] += 1
                    else:
                        obj.low_other[j] += 1

    def compute_low_ratio(self, objects): # 下位行動の割合を算出
        for obj in objects:
            count = len(obj.bboxes)
            for i in range(len(obj.low_posture)):
                obj.low_posture[i] /= count
            for i in range(len(obj.low_body_dir)):
                obj.low_body_dir[i] /= count
            for i in range(len(obj.low_upper_slope)):
                obj.low_upper_slope[i] /= count
            for i in range(len(obj.low_upper_move)):
                obj.low_upper_move[i] /= count
            for i in range(len(obj.low_hand_pos)):
                obj.low_hand_pos[i] /= count
            for i in range(len(obj.low_hand_move)):
                obj.low_hand_move[i] /= count
            for i in range(len(obj.low_face_dir)):
                obj.low_face_dir[i] /= count
            for i in range(len(obj.low_face_slope)):
                obj.low_face_slope[i] /= count
            for i in range(len(obj.low_face_move)):
                obj.low_face_move[i] /= count
            for i in range(len(obj.low_eye_target)):
                obj.low_eye_target[i] /= count
            for i in range(len(obj.low_hand_hold)):
                obj.low_hand_hold[i] /= count
            for i in range(len(obj.low_motion)):
                obj.low_motion[i] /= count
            for i in range(len(obj.low_other)):
                obj.low_other[i] /= count

    def make_actlable_col(self, objects): # 多クラス分類用のデータ変換
        self.count_low(objects) # 下位クラスをカウント
        self.compute_low_ratio(objects) # 下位クラスの割合を算出
        for obj in objects:
            obj.high = obj.acts[-1] # 上位アクティビティに，上位行動IDを格納
            obj.low_all = obj.low_posture + obj.low_body_dir + obj.low_upper_slope + obj.low_upper_move + obj.low_hand_pos + obj.low_hand_move + obj.low_face_dir + obj.low_face_slope + obj.low_face_move + obj.low_eye_target + obj.low_hand_hold + obj.low_motion + obj.low_other

    def expand_list(self, df):
        for column in df:
            if isinstance(df[column].iloc[0], list):
                df = df.drop(column, axis=1).join(df[column].apply(pd.Series).rename(columns=lambda x: f"{column}_{x}"))
        return df

    def processing(self, df_gt):
        result = {
            "image":[],
            "person":[],
            "high_activity":[],
            "num_detected":[],
            "bbox": [],
            "low_action":[]
        }
        col_name = ["image", "person", "high_activity", "num_detected", "min_x", "min_y", "max_x", "max_y"]
        col_name += self.l_action_names

        imgs = list(df_gt.groupby(0).groups.keys())
        for img in imgs:
            df_img = df_gt.groupby(0).get_group(img) # 動画ごとに抽出

            objects = [] # Initialize target objects list
            frms = list(df_img.groupby(1).groups.keys())
            for frm in frms:
                df_frm = df_img.groupby(1).get_group(frm) # フレームごとに抽出

                frm_person_ids = list(df_frm.groupby(7).groups.keys())
                if frm > 0: # bboxes in previous frame
                    prev_bboxes = curr_bboxes
                curr_bboxes = [] # bboxes in current frame: min_x, min_y, max_x, max_y
                frm_action_labels = [] # 上位アクティビティ
                frm_low_actions = [] # 下位アクティビティ
                for frm_person_id in frm_person_ids:
                    print(df_frm)
                    df_person = df_frm.groupby(7).get_group(frm_person_id) # personごとに抽出
                    person_s = df_person.sort_values(6, ascending=True) # sort

                    bbox = [person_s.iloc[0][2], person_s.iloc[0][3], person_s.iloc[0][4], person_s.iloc[0][5]]
                    all_action = person_s[6].unique() 
                    high = list(filter(lambda x: x<=9, all_action)) # 上位アクティビティ
                    low = list(filter(lambda x: x>=10, all_action)) # 下位アクティビティ

                    curr_bboxes.append(bbox)
                    if len(high) > 0:
                        frm_action_labels.append(high[0])
                    else:
                        frm_action_labels.append(-1)
                    frm_low_actions.append(low)
                    #print(img, frm, frm_person_id, bbox, high[0], low)
                
                objects = self.update_objects(objects, curr_bboxes, frm_action_labels, frm_low_actions) # Update objects list

            #print(img, frm)
            objects = [obj for obj in objects if len(obj.bboxes) > self.frm_thresh] # Exclude rare objects
            objects = [obj for obj in objects if obj.acts[-1] > 0] # High_activity > 0
            self.get_average_bbox(objects)

            #if self.multi_cls:# 多クラス分類
            self.make_actlable_col(objects)

            for obj in objects:
                result["image"].append(img)
                result["person"].append(obj.id)
                result["high_activity"].append(obj.high)
                result["num_detected"].append(len(obj.bboxes))
                result["bbox"].append(obj.avg_bbox)
                result["low_action"].append(obj.low_all)

        df = pd.DataFrame(result)
        df = self.expand_list(df)
        df.columns = col_name
        return df

if __name__ == '__main__':
    args = parse_args()
    if args.config is None:
        import conf
        config = conf
    else:
        from mmengine.config import Config
        config = Config.fromfile(args.config)

    converter = ConvertRF(config)
    converter(config.anno_train_csv, config.rf_train_csv)
    #converter(config.anno_val_csv, config.rf_val_csv)
