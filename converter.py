#####
# 目的：画像のファイル名を*.jpgからimg_00000.jpgに変換
#####


import os
import numpy as np
import cv2
import glob

DATA_DIR = 'KOKUYO_data/IMG_1819'
video_id = "IMG_1819"

OUT_PATH = 'KOKUYO_data/convert'
os.makedirs(OUT_PATH, exist_ok=True)

def convert_img_filename():
    #フォルダ名を指定
    files = glob.glob(os.path.join(DATA_DIR, '*.jpg'))
    files.sort()
    #全てのファイルを実行
    for i,f in enumerate(files):
        
        # iを左詰めの数列に変更　ex) i=1 → s=00001
        s = f'{i}'
        a = s.zfill(5)
        # 画像ファイルを読み込む
        img = cv2.imread(f)

        # 変換した値で保存
        cv2.imwrite(os.path.join(OUT_PATH, f"img_{a}.jpg"),img)

def convert_spatio_label():
    #####
    # 目的：Spatio-temporal action recovnition用のラベルファイルに変換
    #####

    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    files.sort()

    # train.txtにファイル書き込み
    with open(os.path.join(OUT_PATH, "train.txt"),"a") as ff:
        for j,file in enumerate(files):
            #ファイル読み込み
            f = open(file, 'r')
            #初期値
            a = -1
            #人の識別用ＩＤ
            person_id = 0

            # 1つのファイルに格納されているファイルを一行ずつ返還
            for i,data in enumerate(f):
                # tmpにリストとして格納
                tmp = list(map(float,data.split()))
                # print(tmp)
                if tmp[0] < 80:
                    # 前のラベルの人物と同一人物か
                    if tmp[0] > a:
                        # ファイル書き込み (video_id,frame,yolo_coorsinate,actionID,personID)
                        ff.write(f"{video_id},{j},{tmp[1]},{tmp[2]},{tmp[3]},{tmp[4]},{int(tmp[0]+1)},{person_id}\n")
                        a = tmp[0]
                    else:
                        person_id  += 1
                        # ファイル書き込み (video_id,frame,yolo_coorsinate,actionID,personID)
                        ff.write(f"{video_id},{j},{tmp[1]},{tmp[2]},{tmp[3]},{tmp[4]},{int(tmp[0]+1)},{person_id}\n")
                        a = -1
        
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
    convert_img_filename()
    convert_spatio_label()