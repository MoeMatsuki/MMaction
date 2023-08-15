import setup as setup
import os
import pickle
import argparse
import pandas as pd
import numpy as np
from common import load_label_map
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from statistics import mean
from sklearn.ensemble import RandomForestClassifier

def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('--config', default=None, help='train config file path')
    parser.add_argument('--RFmodel', default="/home/moe/MMaction/config/clf_model_WB.pkl")
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

class mm2Act:
    def __init__(self, config, rfmodel):
        if rfmodel is not None: # RFモデル
            self.rf_model =  rfmodel

        # 上位行動ラベル
        self.action_label_path = config.action_label
        # 下位行動ラベル
        self.label_map = config.label_map

        # 対象として扱う出現フレーム数の閾値
        self.frm_thresh = config.mm_thresh

        # 多クラス分類かどうか
        self.multi_cls = config.flag_multi_cls
        # ２値の場合のTrueとする行動IDのリスト
        self.true_ids = config.true_ids

        # このidを含むサンプルは除外される
        self.exclude_sample_ids = config.exclude_sample_ids
        self.set_label()

    def run_inference(self, df_img):
        # モデルの読み込み
        with open(self.rf_model, 'rb') as f:
            clf = pickle.load(f)

        # 推論結果を格納する空のリストを作成
        results = []

        # データフレームの各行に対して推論を実行
        for index, row in df_img.iterrows():
            data = row[8:]  # 8列目以降のデータを取得
            result = clf.predict(np.array(data).reshape(1, -1))  # 入力データを1行の2次元配列に変換
            results.append(result[0])  # 結果リストに推論結果を追加
            
        return results

    def integrate_solo(self, df_img):
        org = df_img['high_activity']
        intg = []
        for i in range(len(org)):
            if org[i] == 5:
                intg.append(2)
            else:
                intg.append(org[i])
        return intg

    def infer_high_act(self, df_img):
        team_flag = self.run_inference(df_img) # RF: team (3, 4) -> 1, solo (1, 2, 5) -> 0
        true_values = self.integrate_solo(df_img)
        high_acts = []

        for i in range(len(df_img)): # for each person
            if team_flag[i] == 1: # team (3, 4)
                wb_act = [] # whiteboard related actions
                wb_act.append(df_img.iloc[i]['hand near whiteboard'])
                wb_act.append(df_img.iloc[i]['wrist moving (writing on a whiteboard)'])
                wb_act.append(df_img.iloc[i]['wrist moving (erase on a whiteboard)'])
                wb_act.append(df_img.iloc[i]['wrist put sticky note on wallpaper'])
                wb_act.append(df_img.iloc[i]['eyes looking (whiteboard)'])
                wb_act.append(df_img.iloc[i]['eyes looking (display)'])
                wb_act.append(df_img.iloc[i]['hand holding (pen)'])
                wb_act.append(df_img.iloc[i]['hand holding (whiteboard eraser)'])
                if sum(wb_act) > 0:
                    high_acts.append(3) # チームシンキング
                else:
                    high_acts.append(4) # チームビルディング (対面mtg)
            else: # solo (1, 2, 5)
                if df_img.iloc[i]['conversation'] > 0: # 会話あり or なし
                    high_acts.append(1) # 参加型Webmtg
                else:
                    high_acts.append(2) # ソロワーク including 聴講型Webmtg
        return high_acts

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

    def process_frame_string(self, frame_str):
        # imgの位置を探す
        img_index = frame_str.rfind("img_")

        # img以前の文字列を取得
        prefix = frame_str[:img_index + 4] # img_の文字列も含めるために+4しています

        # .jpgの位置を探す
        jpg_index = frame_str.rfind(".jpg")

        # .jpg以前とimg以降の文字列（つまり、ID部分）を取得
        id_str = frame_str[img_index + 4: jpg_index] # imgの文字列を除くために+3しています

        # 文字列形式のIDを整数に変換
        frm = int(id_str)

        return prefix, frm

    def transformMM(self, df):
        reverse_l_action_label = {v: k for k, v in self.l_action_label.items()}
        df['prefix'], df['id'] = zip(*df['frame'].apply(self.process_frame_string))
        df['counter'] = df.groupby('id').cumcount() # フレームIDごとにカウンタを生成

        # 列7から80までについて閾値以上の値の列名を取得
        th = 0.5  # 閾値を設定
        columns = df.columns[6:78]  # 列名リスト
        activities = df[columns].apply(lambda x: [reverse_l_action_label[col] for col in x.index if x[col] >= th and col in reverse_l_action_label], axis=1)
        df['activities'] = activities

        df_dt = pd.DataFrame()
        rows_list = []
        for index, row in df.iterrows():
            prefix, id, min_x, min_y, max_x, max_y, activities, counter = row['prefix'], row['id'], row['min_x'], row['min_y'], row['max_x'], row['max_y'], row['activities'], row['counter']

            # 各アクティビティに対して行を追加
            for activity in activities:
                new_row = {'prefix': prefix, 'id': id, 'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y, 'activity': activity, 'counter': counter}
                rows_list.append(new_row)
        
        df_new = pd.DataFrame(rows_list)
        df_dt = pd.concat([df_dt, df_new], ignore_index=True)

        return df_dt

    def __call__(self, input, output):
        df = pd.read_csv(input, header=0)
        train_df = self.transformMM(df)
        train_df = self.processing(train_df)
        train_df.to_csv(output, index=False)

    def has_duplicates(self, seq):
        return len(seq) != len(set(seq))

    def compute_distance(self, bbox1, bbox2):
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    def traceObjects(self, prev_objects, curr_bboxes, curr_act):
        max_frames_not_seen = 30 # threshold for identification
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

        imgs = list(df_gt.groupby('prefix').groups.keys())
        for img in imgs:
            df_img = df_gt.groupby('prefix').get_group(img) # 動画ごとに抽出

            objects = [] # Initialize target objects list
            frms = list(df_img.groupby('id').groups.keys())
            for frm in frms:
                df_frm = df_img.groupby('id').get_group(frm) # フレームごとに抽出

                frm_person_ids = list(df_frm.groupby('counter').groups.keys())
                if 'curr_bboxes' in locals(): # bboxes in previous frame
                    prev_bboxes = curr_bboxes
                curr_bboxes = [] # bboxes in current frame: min_x, min_y, max_x, max_y
                frm_action_labels = [] # 上位アクティビティ
                frm_low_actions = [] # 下位アクティビティ
                for frm_person_id in frm_person_ids:
                    df_person = df_frm.groupby('counter').get_group(frm_person_id) # personごとに抽出
                    person_s = df_person.sort_values('activity', ascending=True) # sort

                    bbox = [person_s.iloc[0][2], person_s.iloc[0][3], person_s.iloc[0][4], person_s.iloc[0][5]]
                    all_action = person_s['activity'].unique() 

                    high = list(filter(lambda x: x<=9, all_action)) # 上位アクティビティ
                    low = list(filter(lambda x: x>=10, all_action)) # 下位アクティビティ

                    curr_bboxes.append(bbox)
                    if len(high) > 0:
                        frm_action_labels.append(high[0])
                    else:
                        frm_action_labels.append(-1)
                    frm_low_actions.append(low)
                    # print(img, frm, frm_person_id, bbox, frm_action_labels[0], low)
                
                objects = self.update_objects(objects, curr_bboxes, frm_action_labels, frm_low_actions) # Update objects list
            print(img, frm)

            objects = [obj for obj in objects if len(obj.bboxes) > self.frm_thresh] # Exclude rare objects
            # objects = [obj for obj in objects if obj.acts[-1] > 0] # High_activity > 0
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

        # infer high act.
        estimated_act = self.infer_high_act(df)
        df['high_activity'] = estimated_act

        return df

def process_dirs(conf_rf, RFmodel, out_csv, dir):
    predict_rf = mm2Act(conf_rf, RFmodel)
    for curDir, _, files in os.walk(dir):
        if "test_mmaction.csv" in files:
            print(curDir)
            csv_path = os.path.join(curDir, "test_mmaction.csv")
            out_csv_path = os.path.join(curDir, out_csv)
            predict_rf(csv_path, out_csv_path)

if __name__ == '__main__':
    args = parse_args()
    if args.config is None:
        from config import conf_rf
        config = conf_rf
    else:
        from mmengine.config import Config
        config = Config.fromfile(args.config)

    # ## Case1: １つずつ
    # mm_csv = "/home/moe/MMaction/data/20230703_サンプル/mm_result/Act01_230703_sample_2023-07-03.zip/test_mmaction.csv"
    # mm2RF_csv = "/home/moe/MMaction/data/20230703_サンプル/mm_result/Act01_230703_sample_2023-07-03.zip/rf_result.csv"
    # converter = mm2Act(config, args.RFmodel)
    # converter(mm_csv, mm2RF_csv)

    ## Case2: まとめて
    out_csv = "rf_result.csv"
    dir = "/home/moe/MMaction/data/2022年8月｜6F改修前/mm_result"
    process_dirs(config, args.RFmodel, out_csv, dir)