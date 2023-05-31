#####
# 目的：画像のファイル名を*.jpgからimg_00000.jpgに変換
#####

import argparse
import os
import numpy as np
import cv2
import glob
from mmcv import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args

def convert_img_filename(files, out_dir):
    #並び替え
    files.sort()
    #全てのファイルを実行
    for i,f in enumerate(files):
        f_base = os.path.basename(f)
        # iを左詰めの数列に変更　ex) i=1 → s=00001
        s = f'{i}'
        a = s.zfill(5)
        # 画像ファイルを読み込む
        img = cv2.imread(f)
        # 変換した値で保存
        cv2.imwrite(os.path.join(out_dir, f"img_{a}.jpg"),img)

def convert_spatio_label(ff, files, video_id):
    #####
    # 目的：Spatio-temporal action recovnition用のラベルファイルに変換
    #####
    files.sort()
    for j,file in enumerate(files):
        f_base = os.path.basename(file)
        # if "Webmtg" in f_base or "IMG" in f_base:
        #ファイル読み込み
        f = open(file, 'r')
        #人の識別用ＩＤ
        person_id = -1

        # 1つのファイルに格納されているファイルを一行ずつ返還
        for i,data in enumerate(f):
            # tmpにリストとして格納
            tmp = list(map(float,data.split()))

            # 前のラベルの人物と同一人物か
            if int(tmp[0]) == 0:
                person_id += 1
                continue
            else:
                min_x = tmp[1] - (tmp[3] / 2)
                min_y = tmp[2] - (tmp[4] / 2)
                max_x = tmp[1] + (tmp[3] / 2)
                max_y = tmp[2] + (tmp[4] / 2)
                # ファイル書き込み (video_id,frame,yolo_coorsinate,actionID,personID)
                ff.write(f"{video_id},{str(j).zfill(4)},{round(min_x, 3)},{round(min_y, 3)},{round(max_x, 3)},{round(max_y, 3)},{int(tmp[0])},{person_id}\n")
        f.close()

if __name__ == '__main__':
    args = parse_args()
    print(args.config)
    cfg = Config.fromfile(args.config)
    print(cfg)

    IMG_DIR = cfg.TRAIN_IMG_DIR#'KOKUYO_data/train_data'
    TXT_DIR = cfg.TRAIN_TXT_DIR#'KOKUYO_data/train_data'
    OUT_PATH = cfg.data_root#'KOKUYO_data/convert_img'
    train_csv = cfg.anno_train_csv
    val_csv = cfg.anno_val_csv
    val_videos = cfg.val_videos

    with open(val_csv,"a") as f_train:
        label_dirs = os.listdir(TXT_DIR)
        print(f"Tere are {label_dirs} in the annotation directory.")
        for label_dir in label_dirs:
            label_dir_path = os.path.join(TXT_DIR, label_dir)
            img_dirs = os.listdir(label_dir_path)
            print(f"Tere are {img_dirs} in the {label_dir}.")
            for img_dir in img_dirs:
                if img_dir in val_videos:
                    print(f"{img_dir} is for val")
                    img_dir_path = os.path.join(label_dir_path, img_dir)
                    frame_files = os.listdir(img_dir_path)
                    annotation_txt_files = glob.glob(os.path.join(img_dir_path, '*.txt'))
                    # 画像が存在するかどうかを確認
                    img_dir_path = os.path.join(IMG_DIR, os.path.join(label_dir, img_dir))
                    assert os.path.isdir(img_dir_path), f"{img_dir_path}が存在しないです。"
                    img_files = glob.glob(os.path.join(img_dir_path, '*.jpg'))
                    # 画像とアノテーションのファイル数が同じかどうか
                    img_len = len(img_files); anno_len = len(annotation_txt_files)
                    assert img_len == anno_len, f"{img_dir_path}の画像とアノテーションのファイル数が一致しません。画像ファイル数：{img_len}、アノテーションファイル数：{anno_len}。"

                    # 画像のファイル名を変更
                    out_put = os.path.join(OUT_PATH, img_dir)
                    os.makedirs(out_put, exist_ok=True)
                    convert_img_filename(img_files, out_put)

                    # train.txtにファイル書き込み                
                    convert_spatio_label(f_train, annotation_txt_files, img_dir)

                    print(f"done {img_dir_path} processing: frame number is {anno_len}")

    with open(train_csv,"a") as f_train:
        label_dirs = os.listdir(TXT_DIR)
        print(f"Tere are {label_dirs} in the annotation directory.")
        for label_dir in label_dirs:
            label_dir_path = os.path.join(TXT_DIR, label_dir)
            img_dirs = os.listdir(label_dir_path)
            print(f"Tere are {img_dirs} in the {label_dir}.")
            for img_dir in img_dirs:
                if img_dir in val_videos:
                    continue
                img_dir_path = os.path.join(label_dir_path, img_dir)
                frame_files = os.listdir(img_dir_path)
                annotation_txt_files = glob.glob(os.path.join(img_dir_path, '*.txt'))
                # 画像が存在するかどうかを確認
                img_dir_path = os.path.join(IMG_DIR, os.path.join(label_dir, img_dir))
                if not os.path.isdir(img_dir_path):
                    img_dir_path = os.path.join(IMG_DIR, img_dir)
                assert os.path.isdir(img_dir_path), f"{img_dir_path}が存在しないです。"
                img_files = glob.glob(os.path.join(img_dir_path, '*.jpg'))
                # 画像とアノテーションのファイル数が同じかどうか
                img_len = len(img_files); anno_len = len(annotation_txt_files)
                if img_len != anno_len:
                    print(f"{img_dir_path}の画像とアノテーションのファイル数が一致しません。画像ファイル数：{img_len}、アノテーションファイル数：{anno_len}。")
                    continue

                # 画像のファイル名を変更
                out_put = os.path.join(OUT_PATH, img_dir)
                os.makedirs(out_put, exist_ok=True)
                convert_img_filename(img_files, out_put)

                # train.txtにファイル書き込み                
                convert_spatio_label(f_train, annotation_txt_files, img_dir)

                print(f"done {img_dir_path} processing: frame number is {anno_len}")