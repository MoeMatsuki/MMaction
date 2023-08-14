import glob
import os
from os import makedirs
import cv2
import json
from os.path import splitext, dirname, basename, join

class Convverter:
    def __init__(self):
        pass

    def vid2img(self, video_path: str, frame_dir: str, 
                name="img", ext="jpg", max_frame_num=float("inf")):
        # ビデオを読み込む
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"cannot open {video_path}")
            return None
        
        # prepare save path name
        frame_dir_ = self.make_dir(video_path, frame_dir)
        base_path = join(frame_dir_, name)

        idx = 0
        count = 0
        while cap.isOpened():
            idx += 1
            ret, frame = cap.read()
            if ret and count < max_frame_num:
                if cap.get(cv2.CAP_PROP_POS_FRAMES) == 1:  # 0秒のフレームを保存
                    cv2.imwrite("{}_{}.{}".format(base_path, "000000", ext),
                                frame)
                elif idx < cap.get(cv2.CAP_PROP_FPS):
                    continue
                else:  # 1秒ずつフレームを保存
                    second = int(cap.get(cv2.CAP_PROP_POS_FRAMES)/idx)
                    filled_second = str(second).zfill(6)
                    cv2.imwrite("{}_{}.{}".format(base_path, filled_second, ext),
                                frame)
                    idx = 0
                count += 1
            else:
                break

    def make_dir(self, path, frame_dir):
        """make dir named video name
        """
        v_name = splitext(basename(path))[0]
        if frame_dir[-1:] == "\\" or frame_dir[-1:] == "/":
            frame_dir = dirname(frame_dir)
        frame_dir_ = join(frame_dir, v_name)
        print(f"{frame_dir_} is saved directory")

        makedirs(frame_dir_, exist_ok=True)
        return frame_dir_

    def convert_img_filename(self, img_dir_path, out_dir):
        files = glob.glob(os.path.join(img_dir_path, '*.jpg'))
        #並び替え
        files.sort()
        #全てのファイルを実行
        for i,f in enumerate(files):
            # iを左詰めの数列に変更　ex) i=1 → s=00001
            s = f'{i}'
            a = s.zfill(6)
            # 画像ファイルを読み込む
            img = cv2.imread(f)
            # 変換した値で保存
            cv2.imwrite(os.path.join(out_dir, f"img_{a}.jpg"),img)

    def normalize(self, bbox, w, h):
        """bboxの正規化
        """
        return [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]

    def save_txt(self, txt_list, file_path):
        text_file = open(file_path, "wt")

        for t in txt_list:
            txt = ",".join(t)
            text_file.write(txt)
            text_file.write("\n")

        text_file.close()

    def json2csv(self, files, video_id):
        files.sort()
        result = []
        for j,file in enumerate(files):
            #ファイル読み込み
            with open(file) as f:
                json_ = json.load(f)
            w = json_["width"]
            h = json_["height"]
            annotations = json_["annotations"]
            for h_ix, human in enumerate(annotations):
                attribute = human["attributes"]
                point = [str(round(i, 3)) for i in self.normalize(human["points"], w, h)]
                for a in attribute:
                    value = a["value"]
                    type = a["type"]
                    if type == "checkbox":
                        for activity_id in value:
                            # 行動クラスの書き込み
                            line = [video_id]
                            line.append(str(j).zfill(6))
                            line.extend(point)
                            line.append(activity_id)
                            line.append(str(h_ix))
                            result.append(line)
                    if type == "select":
                        # ラベルが存在しないものはスキップ
                        if value == "":
                            continue
                        # 行動クラスの書き込み
                        line = [video_id]
                        line.append(str(j).zfill(6))
                        line.extend(point)
                        line.append(value)
                        line.append(str(h_ix))
                        result.append(line)
        return result

    def txt2csv(self, files, video_id):
        #####
        # 目的：Spatio-temporal action recovnition用のラベルファイルに変換
        #####
        files.sort()
        result = ""
        for j,file in enumerate(files):
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
                    result = result +f"{video_id},{str(j).zfill(6)},{round(min_x, 3)},{round(min_y, 3)},{round(max_x, 3)},{round(max_y, 3)},{int(tmp[0])},{person_id}\n"
            f.close()
            # if "IMG_1817" in file and j == 9:
            #         print(file)
            #         print(result)
        return result

    def save_label_csv(self, dirs, out_csv):
        res = ""
        list_res = []
        for curDir, _, _ in os.walk(dirs):
            annotation_json_files = glob.glob(os.path.join(curDir, '*.json'))
            annotation_txt_files = glob.glob(os.path.join(curDir, '*.txt'))
            video_name = curDir.split("/")[-1]
            if len(annotation_txt_files) > 5:
                res = res + self.txt2csv(annotation_txt_files, video_name)
            if len(annotation_json_files) > 5:
                list_res.extend(self.json2csv(annotation_json_files, video_name))
        if len(res) != 0:
            with open(out_csv,"w") as f:
                f.write(res)
        else:
            self.save_txt(list_res, out_csv)

def main():
    con = Convverter()

    # ## Case1: videoからimgに変換
    # dir = "/home/moe/MMaction/data/230623_実験前テスト/original"
    # save_dir = "/home/moe/MMaction/data/230623_実験前テスト/convert_img"
    # vids = glob.glob(f'{dir}/*.mp4')
    # for vid in vids:
    #     con.vid2img(vid, save_dir)
    #     print(f"{vid} is done")

    ## Case2-1: 任意のディレクトリから画像を抽出して、名前を変換したものを別のディレクトリに保存
    img_dir = "/home/moe/MMaction/fastlabel1/train_img/Webmtg_221226_02"
    save_dir = "/home/moe/MMaction/fastlabel1/convert_img/Webmtg_221226_02"
    os.makedirs(save_dir, exist_ok=True)
    con.convert_img_filename(img_dir, save_dir)

    # ## Case2-2: まとめて
    # input_dir = "/home/moe/MMaction/fastlabel1/train_img/"
    # save_dir = "/home/moe/MMaction/fastlabel1/convert_img/"
    # img_dirs = os.listdir(input_dir)
    # for img_dir in img_dirs:
    #     # if img_dir in val_videos:
    #     print(img_dir)
    #     img_dir_path = os.path.join(input_dir, img_dir)
    #     assert os.path.isdir(img_dir_path), f"{img_dir_path}が存在しないです。"
        
    #     out_put = os.path.join(save_dir, img_dir)
    #     os.makedirs(out_put, exist_ok=True)
    #     con.convert_img_filename(img_dir_path, out_put)

    ## Case3-1: labelのテキストファイルを抽出して一つのcsvにまとめる
    dir_path = "/home/moe/MMaction/fastlabel1/annotation"
    out_csv = "annotation_test.csv"
    con.save_label_csv(dir_path, out_csv)

    # ## Case3-2: jsonにも対応
    # dir_path = "/home/moe/MMaction/fastlabel1/jsons"
    # out_csv = "annotation_json_test.csv"
    # con.save_label_csv(dir_path, out_csv)

if __name__ == "__main__":
    main()
