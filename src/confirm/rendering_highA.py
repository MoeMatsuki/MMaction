import cv2
import copy
import numpy as np
import os
import pandas as pd
from natsort import natsorted
from rendering import rendering_bboxes


class CheckData:
    def __init__(self, label_top_n=4):
        self.label_top_n = label_top_n

    def check_img(self, img_names, test_video_dir):
        """
        csvに存在する画像が手元にあるかどうかの確認
        """
        flag_continue = True
        for img_name in img_names:
            if not os.path.exists(os.path.join(test_video_dir, img_name)):
                print(f"{img_name} is not exist")
                flag_continue = False
        return flag_continue

    def convert_img_name(self, df, test_video_dir):
        f = lambda x: "img_" + str(x).zfill(6)
        df_ = copy.deepcopy(df)
        df_.loc[:, "file_name"] = df[1].map(f)
        df_.loc[:, "path"] = test_video_dir +  df[0] + "/" + df_["file_name"] + ".jpg"
        return df_

    def check_as_images(self, df_, test_video_dir, out_dir):
        img_names = df[0].unique()
        for img_name in img_names:
            print(img_name)
            # 画像のファイル名を取得
            test_video = os.path.join(test_video_dir, img_name)
            img_name_files = natsorted(os.listdir(test_video))

            # 画像のファイル名に合わせる
            d = df_[df_[0] == img_name]
            d = self.convert_img_name(d, test_video_dir)

            # 出力ディレクトリ
            out_img_dir = os.path.join(out_dir, img_name)
            os.makedirs(out_img_dir, exist_ok=True)

            for frame_file in img_name_files:
                print(frame_file)
                frame_file_base = frame_file.split(".")[0]
                d_frame = d[d["file_name"] == frame_file_base]
                if len(d_frame) == 0:
                    print(frame_file_base, d["file_name"])
                    assert ValueError
                bboxes, labels_list = self.get_info(d_frame)
                img_path = d_frame.loc[:, "path"].values[0]
                out_img = os.path.join(out_img_dir, frame_file)
                self.save_img(img_path, bboxes, labels_list, out_img)

    def get_info(self, d):
        bboxes = []
        labels_list = []
        for i in d[2].unique():
            # 同じbboxのものでソート
            d_tmp = d[d[2] == i]
            d_tmp = d_tmp.sort_values(6)
            # 行動ラベルが0以上
            d_filter = d_tmp[d_tmp.loc[:, 6] > 0]
            box  = d_filter.iloc[0, [2,3,4,5]].values
            labels = list(d_filter.loc[:, 6].values)
            bboxes.append(box)
            labels_list.append(labels)
        return bboxes, labels_list

    def save_img(self,test_img_path, test_box, labels, out_img_path):
        test_img = cv2.imread(test_img_path)
        res = rendering_bboxes(test_img, test_box, labels, label_top_n=self.label_top_n)
        cv2.imwrite(out_img_path, res)

if __name__ == "__main__":
    cheker = CheckData()

    test_video_dir = "/home/moe/MMaction/fastlabel1/convert_img/"
    train_csv = "/home/moe/MMaction/annotation_test.csv"
    out_dir = "out"

    df = pd.read_csv(train_csv, header=None)
    img_names = df[0].unique()

    # Case1: csvに存在する画像が手元にあるかどうかの確認
    is_images = cheker.check_img(img_names, test_video_dir)
    print(is_images)

    # Case2: bboxとラベルを描写
    cheker.check_as_images(df, test_video_dir, out_dir)

