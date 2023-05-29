"""convert from train.csv to test_rf.csv for imprement randomforest
"""

import pandas as pd
import mmcv
import argparse
from mmcv import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--RFmodel', default=None)
    args = parser.parse_args()
    return args

class ConvertRF:
    def __init__(self, config):
        self.col_name = ["img_name", "frame_sec", "min_x", "min_y", "max_x", "max_y", "action_id", "person_id"]
        self.train_csv = config.anno_train_csv#"/home/moe/MMaction/KOKUYO_data/annotations/train.csv"
        self.TRAIN_CSV = config.rf_train_csv
        self.val_csv = config.anno_val_csv#"/home/moe/MMaction/KOKUYO_data/annotations/train.csv"
        self.VAL_CSV = config.rf_val_csv
        self.action_label_path = config.action_label
        self.config = config
        self.multi_cls = config.multi_cls
        self.true_ids = config.true_ids

        self.action_label = self.load_label_map(self.action_label_path)
        self.action_label_ids = list(self.action_label.keys())

        self.label_map = self.get_label()
        label_ids = list(self.label_map.keys())
        if len(self.true_ids) != 0:
            for true_id in self.true_ids:
                if true_id in self.action_label_ids:
                    continue
                print("exclude samples")
                print(self.label_map[true_id])
                del self.label_map[true_id]

        self.exclude_sample_ids = config.exclude_sample_ids

    def load_label_map(self, file_path):
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
    def get_label(self):
        label_map = self.load_label_map(self.config["label_map"])
        try:
            if self.config['data']['train']['custom_classes'] is not None:
                label_map = {
                    # id + 1: label_map[cls]
                    cls: label_map[cls]
                    for id, cls in enumerate(self.config['data']['train']
                                            ['custom_classes'])
                }
        except KeyError:
            pass
        return label_map

    def __call__(self):
        train_df = pd.read_csv(self.train_csv, header=None)
        train_df = self.processing(train_df)
        train_df.to_csv(self.TRAIN_CSV)

        val_df = pd.read_csv(self.val_csv, header=None)
        val_df = self.processing(val_df)
        val_df.to_csv(self.VAL_CSV)

    def make_binary_col(self, frame_result, count, action_ids):
        action_ids_list = action_ids.values.tolist()
        frame_result["action_id"][count] = action_ids_list
        for id in self.true_ids:
            if id in action_ids_list:
                frame_result["action_label"][count] = 1
                return None
        frame_result["action_label"][count] = 0

        for ix in action_ids_list:
            if (ix in self.action_label_ids) or (ix in self.true_ids):
                continue
            action_name = self.label_map[ix]
            frame_result[action_name][count] = 1

        for id in self.exclude_sample_ids:
            if id in action_ids_list:
                frame_result["action_label"][count] = None

    def make_actlable_col(self, frame_result, count, action_id):
        if action_id in self.action_label_ids:
            frame_result["action_id"][count] = action_id
            frame_result["action_label"][count] = self.action_label[action_id]
            return None
        action_name = self.label_map[action_id]
        if action_name in self.labels:
            frame_result[action_name][count] = 1

    def processing(self, df_gt):
        df_gt = df_gt.set_axis(self.col_name, axis='columns')
        convert_str = lambda x: str(x).zfill(5)
        df_gt["id"] = df_gt["img_name"] + "_" + df_gt["frame_sec"].map(convert_str) + "_" + df_gt["person_id"].map(convert_str)
        group = df_gt.groupby("id")
        frame_result = {
            "frame":[],
            "min_x": [],
            "min_y": [],
            "max_x": [],
            "max_y": [],
            "action_label":["NaN"] * len(group.groups),
            "action_id":[0] * len(group.groups)
        }
        [frame_result.update({l: [0] * len(group.groups)}) for l in self.label_map.values()]
        count = 0
        for id, ix_list in group.groups.items():
            frame_result["frame"].append(id)
            l = df_gt.iloc[ix_list[0], :]
            frame_result["min_x"].append(l["min_x"])
            frame_result["min_y"].append(l["min_y"])
            frame_result["max_x"].append(l["max_x"])
            frame_result["max_y"].append(l["max_y"])
            if self.multi_cls:
                for ix in ix_list:
                    action_id = df_gt.iloc[ix, 6]
                    self.make_actlable_col(frame_result, count, action_id)
            else:
                action_ids = df_gt.iloc[list(ix_list), 6]
                self.make_binary_col(frame_result, count, action_ids)
            count += 1
        # for k,v in frame_result.items():
        #     print(k, len(v))
        return pd.DataFrame(frame_result)
    
if __name__ == '__main__':
    args = parse_args()
    config = Config.fromfile(args.config)
    if args.RFmodel is not None:
        config.rf_model =  args.RFmodel
    # df = load_csv("test_mmaction2.csv")

    converter = ConvertRF(config)
    converter()

