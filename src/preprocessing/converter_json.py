import json
import os
import cv2
import argparse
from mmengine.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args

def normalize(bbox, w, h):
    return [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]

def make_list(json_, w, h):
    w = json_["width"]
    h = json_["height"]
    annotations = json_["annotations"]
    result = []
    for human in annotations:
        attribute = human["attributes"]
        point = [str(i) for i in normalize(human["points"], w, h)]
        line = ["0"]; line.extend(point)
        result.append(line)
        for a in attribute:
            value = a["value"]
            if isinstance(value, list):
                for i in value:
                    line = [i]; line.extend(point)
                    result.append(line)
            if isinstance(value, str):
                if value == "":
                    continue
                line = [value]; line.extend(point)
                result.append(line)
    return result

def save_txt(txt_list, file_path):
    text_file = open(file_path, "wt")

    for t in txt_list:
        txt = " ".join(t)
        text_file.write(txt)
        text_file.write("\n")

    text_file.close()

def main():
    dir = "/home/moe/MMaction/fastlabel1/json/"
    out_dir = "/home/moe/MMaction/fastlabel1/annotation/"
    # args = parse_args()
    # cfg = Config.fromfile(args.config)

    # trainimg_dir = cfg.TRAIN_IMG_DIR#'KOKUYO_data/train_data'
    # dir = cfg.JSON_DIR#'KOKUYO_data/train_data'
    # out_dir = cfg.TRAIN_TXT_DIR#'KOKUYO_data/convert_img'

    for label in os.listdir(dir):
        label_dir = os.path.join(dir, label)
        out_label = os.path.join(out_dir, label)
        if os.path.isdir(label_dir):
            for img_name in os.listdir(label_dir):
                img_dir = os.path.join(label_dir, img_name)
                out_img = os.path.join(out_label, img_name)
                os.makedirs(out_img, exist_ok=True)
                if os.path.isdir(img_dir):
                    # 動画名とフレーム数
                    print(img_name, len(os.listdir(img_dir)))
                    for img_file in os.listdir(img_dir):
                        if ".json" in img_file:
                            path = os.path.join(img_dir, img_file)
                            out_file = os.path.join(out_img, img_file.replace(".json", ".txt"))

                            # json読み取り
                            with open(path) as f:
                                json_ = json.load(f)
                            # json読み取り
                            txt_list = make_list(json_, w, h)
                            save_txt(txt_list, out_file)

if __name__ == "__main__":
    main()
