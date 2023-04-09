#####
# 目的：画像のファイル名を*.jpgからimg_00000.jpgに変換
#####


import os
import numpy as np
import cv2
import glob

def convert_img_filename(data_dir, out_dir):
    #フォルダ名を指定
    files = glob.glob(os.path.join(data_dir, '*.jpg'))
    files.sort()
    #全てのファイルを実行
    for i,f in enumerate(files):
        f_base = os.path.basename(f)
        if "Webmtg" in f_base or "IMG" in f_base:
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

    for j,file in enumerate(files):
        f_base = os.path.basename(file)
        if "Webmtg" in f_base or "IMG" in f_base:
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

def convert_actionr_label():
    #####
    # 目的：Action recovnition用のラベルファイルに変換
    #####

    video_id = DATA_DIR
    label_id = 0
    with open("train_A.txt","a") as ff:
        ff.write(f"{video_id}\t{label_id}")

if __name__ == '__main__':
    DATA_DIR = 'KOKUYO_data/train_data'
    OUT_PATH = 'KOKUYO_data/convert'
    ANNO_PATH = 'KOKUYO_data/annotations'

    with open(os.path.join(ANNO_PATH, "train.csv"),"a") as ff:
        for label_dir in os.listdir(DATA_DIR):
            label_d = os.path.join(DATA_DIR, label_dir)
            if os.path.isdir(label_d):
                for img_dir in os.listdir(label_d):
                    data_dir = os.path.join(label_d, img_dir)
                    if os.path.isdir(data_dir):
                        print(data_dir)
                        out_put = os.path.join(OUT_PATH, img_dir)
                        print(out_put)
                        os.makedirs(out_put, exist_ok=True)
                        convert_img_filename(data_dir, out_put)

                        files = glob.glob(os.path.join(data_dir, '*.txt'))
                        files.sort()
                        # train.txtにファイル書き込み
                        convert_spatio_label(ff, files, img_dir)