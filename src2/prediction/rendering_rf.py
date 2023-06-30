from rendering import visualize

import os
import pandas as pd
from natsort import natsorted
import cv2
import moviepy.editor as mpy
import numpy as np

dir = "/home/moe/MMaction/result_slowonly_val"
img_dir = "/home/moe/MMaction/fastlabel1/convert_img"
label_map = {1: "teamBD(wb_mtg)", 2: "solo", 3:"teamTH", 4: "teamBD(F2F)"}
output_fps = 4
box_col = ["min_x","min_y","max_x","max_y"]

def combin_result(mm_df, rf_df):
    rf_bboxes = np.array([calc_center(box) for box in rf_df.loc[:, box_col].values])
    high_activities = rf_df.loc[:, "high_activity"].values
    
    mm_bboxes = np.array([calc_center(box) for box in mm_df.loc[:, box_col].values])
    # mmactionの回答１つづつに最も近いrfの結果を探す
    close_result = search_pair(mm_bboxes, rf_bboxes)
    h_activities = []
    for c in range(len(mm_bboxes)):
        ix = close_result[c][0]
        lb_ix = high_activities[ix]
        h_activities.append(label_map[lb_ix])
        
    frame_imgs = mm_df.loc[:, "frame"].values
    mmboxes = mm_df.loc[:, box_col].values

    h_activity_dict = []
    for i in range(len(h_activities)):
        h_activity_dict.append({
            "img": frame_imgs[i],
            "high_activity": h_activities[i],
            "bbox": mmboxes[i],
        })
    return h_activity_dict

def calc_center(bbox):
    x_center = bbox[0] + (bbox[2] - bbox[0]) / 2
    y_center = bbox[1] + (bbox[3] - bbox[1]) / 2
    return [x_center, y_center]

def search_pair(mm_list, rf_list):
    results = {}
    for i_ix, i in enumerate(mm_list):
        min_ix = 0
        min = float("inf")
        for j_ix, j in enumerate(rf_list):
            c = np.linalg.norm(i - j)
            if min > c:
                min = c
                min_ix = j_ix
            else:
                pass
        results.update({i_ix: (min_ix, min)})
    return results

def main():
    for curDir, dirs, files in os.walk(dir):
        if "rf_result.csv" in files:            
            vid_name = curDir.split("/")[-1]
            print(vid_name)
            out_video = os.path.join(curDir, "rf_result.mp4")

            # データ読み込み
            rf_res_path = os.path.join(curDir, "rf_result.csv")
            mm_res_path = os.path.join(curDir, "test_mmaction.csv")
            rf_df = pd.read_csv(rf_res_path)
            mm_df = pd.read_csv(mm_res_path)

            # mmactionの回答にrfの結果を結合
            result_list = combin_result(mm_df, rf_df)

            frames = []
            results = []
            pre_frame = None
            person_info = []
            new_frame = []
            # 結果をフレームづつのlistに更新
            for ix, frame_info in enumerate(result_list):
                if pre_frame is None:
                    pre_frame = frame_info["img"]
                img_path = os.path.join(img_dir, (os.path.join(vid_name, frame_info["img"]+".jpg")))
                frames.append(cv2.imread(img_path))
                if pre_frame == frame_info["img"]:
                    person_info.append([np.array(frame_info["bbox"]), [frame_info["high_activity"]]])
                else:
                    results.append(person_info)
                    new_frame.append(frames[ix-1])
                    person_info = [[np.array(frame_info["bbox"]), [frame_info["high_activity"]]]]
                pre_frame = frame_info["img"]
            results.append(person_info)
            new_frame.append(frames[ix])

            # 動画に出力
            vis_frames = visualize(new_frame, results)
            vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames],
                                        fps=output_fps)
            vid.write_videofile(out_video)

if __name__ == '__main__':
    main()

    



     
        