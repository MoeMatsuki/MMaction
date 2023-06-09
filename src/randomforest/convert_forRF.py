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
import copy
from common import load_label_map

def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('--config', default=None, help='train config file path')
    args = parser.parse_args()
    return args

class ConvertRF:
    def __init__(self, config):
        self.col_name = ["img_name", "frame_sec", "min_x", "min_y", "max_x", "max_y", "action_id", "person_id"]

        # # csvファイル
        # self.train_csv = config.anno_train_csv#"/home/moe/MMaction/KOKUYO_data/annotations/train.csv"
        # self.TRAIN_CSV = config.rf_train_csv
        # self.val_csv = config.anno_val_csv#"/home/moe/MMaction/KOKUYO_data/annotations/train.csv"
        # self.VAL_CSV = config.rf_val_csv

        # 上位行動ラベル
        self.action_label_path = config.action_label
        # 下位行動ラベル
        self.label_map = config.label_map

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
        train_df.to_csv(output)

        # val_df = pd.read_csv(self.val_csv, header=None)
        # val_df = self.processing(val_df)
        # val_df.to_csv(self.VAL_CSV)

    def has_duplicates(self, seq):
        return len(seq) != len(set(seq))

    def make_binary_col(self, frame_result, count, action_ids):
        """２値クラス分類用のデータ変換

        action_label に1 or 0を格納。true_idsに存在するものが1
        下位行動（説明変数columns）に1 or 0を格納
        """
        action_ids_list = action_ids.values.tolist()
        frame_result["action_id"][count] = copy.copy(action_ids_list)

        # 目的変数を格納
        for id in self.true_ids:
            if id in action_ids_list:
                frame_result["action_label"][count] = 1
                break
            else:
                frame_result["action_label"][count] = 0

        # 説明変数を格納
        for ix in action_ids_list:
            # 上位行動とtrue_idsに含まれる行動は説明変数とならないのでスキップ
            if (ix in self.h_action_label_ids) or (ix in self.true_ids):
                continue

            # id→name
            action_name = self.l_action_label[ix]
            if action_name in self.l_action_names:
                frame_result[action_name][count] = 1

        # 除外するidを含むサンプルの目的変数にNoneを格納
        for id in self.exclude_sample_ids:
            if id in action_ids_list:
                frame_result["action_label"][count] = None

        # 上位行動を含まないサンプルにNoneを格納
        for l in self.h_action_label_ids:
            if l in action_ids_list:
                return None        
        frame_result["action_label"][count] = None

    def make_actlable_col(self, frame_result, count, action_id):
        """多クラス分類用のデータ変換

        action_label に上位行動idを格納
        下位行動（説明変数columns）に1 or 0を格納
        """
        if action_id in self.h_action_label_ids:
            frame_result["action_id"][count] = action_id
            frame_result["action_label"][count] = self.h_action_label[action_id]
            return None
        action_name = self.l_action_label[action_id]
        if action_name in self.l_action_names:
            frame_result[action_name][count] = 1

    def processing(self, df_gt):

        # 一人分のサンプルとするためのユニークなidを定義
        df_gt = df_gt.set_axis(self.col_name, axis='columns')
        convert_str = lambda x: str(x).zfill(5)
        df_gt["id"] = df_gt["img_name"] + "_" + df_gt["frame_sec"].map(convert_str) + "_" + df_gt["person_id"].map(convert_str)
        group = df_gt.groupby("id")

        # 最終的な結果の箱を作成
        frame_result = {
            "frame":[],
            "min_x": [],
            "min_y": [],
            "max_x": [],
            "max_y": [],
            "action_label":["NaN"] * len(group.groups),
            "action_id":[0] * len(group.groups)
        }
        [frame_result.update({l: [0] * len(group.groups)}) for l in self.l_action_label.values()]
        count = 0


        for id, ix_list in group.groups.items():

            frame_result["frame"].append(id)

            # bbox座標を格納
            l = df_gt.iloc[ix_list[0], :]
            frame_result["min_x"].append(l["min_x"])
            frame_result["min_y"].append(l["min_y"])
            frame_result["max_x"].append(l["max_x"])
            frame_result["max_y"].append(l["max_y"])

            if self.multi_cls:# 多クラス分類
                for ix in ix_list:
                    action_id = df_gt.iloc[ix, 6]
                    self.make_actlable_col(frame_result, count, action_id)
            else:# 2値クラス分類
                action_ids = df_gt.iloc[list(ix_list), 6]
                self.make_binary_col(frame_result, count, action_ids)
            count += 1

        return pd.DataFrame(frame_result)
    
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

