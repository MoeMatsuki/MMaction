import pickle
import pandas as pd
import cv2
import os
import argparse
import os.path as osp
import shutil
from common import load_label_map, convert_result, visualize
from convert_forRF import ConvertRF
from natsort import natsorted


test_df = "test_mmaction.csv"
result_rf_path = "result_rf.csv"

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    parser.add_argument('--conf_path', default=None)
    parser.add_argument('--video', default=None)
    parser.add_argument('--input', default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--model', default=None)
    args = parser.parse_args()
    return args

def frame_extraction(video_path):
    """Extract frames given video_path.
    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    print(f"target video is {video_path}")
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    while flag:
        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)
        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()
    return frame_paths, frames

def main():
    args = parse_args()
    if args.conf_path is None:
        import conf
        config = conf
    else:
        from mmengine.config import Config
        config = Config.fromfile(args.conf_path)

    # モデルの読み込み
    if args.model is not None:
        config.rf_model = args.model
    # frames_path, _ = frame_extraction(video)
    with open(config.rf_pickle_path, mode='rb') as f:  # with構文でファイルパスとバイナリ読み来みモードを設定
        model = pickle.load(f)

    # 入力データ
    label_map = load_label_map(config.label_map)
    labels = list(label_map.values())
    if args.input is not None:
        print(os.path.join(args.input, test_df))
        df = pd.read_csv(os.path.join(args.input, test_df))
    else:
        converter = ConvertRF(config)
        converter(config.anno_train_csv, config.rf_train_csv)
        df = pd.read_csv(config.rf_train_csv)
        df = df[df["frame"].str.contains(config.test_video)]
    X = df.loc[:, labels].values 

    y_train_pred = model.predict(X)
    out_df = pd.concat([df, pd.Series(y_train_pred)], axis=1)
    out_df = out_df.rename(columns={0: "pred_action_label"})
    print(out_df)
    if args.out is not None:
        out_df.to_csv(os.path.join(args.out, config.result_rf_path), index = False)
    else:
        os.makedirs(config.out_path, exist_ok=True)
        out_df.to_csv(os.path.join(config.out_path, config.result_rf_path), index = False)



    try:
        import moviepy.editor as mpy
    except ImportError:
        raise ImportError('Please install moviepy to enable output file')

    if args.out is not None:
        frames_path = out_df.loc[:, "frame"].to_list()
        frames_path = list(dict.fromkeys(frames_path))
        frames = [cv2.imread(f) for f in frames_path]
        results = convert_result(out_df, labels)

    else:
        img_dir = os.path.join(config.img_root, config.test_video)
        frames_path = os.listdir(img_dir)
        frames_path = natsorted(frames_path)
        frames = [cv2.imread(os.path.join(img_dir, f)) for f in frames_path]
        results = []
        for ix, f in enumerate(frames):
            h,w = f.shape[:2]
            num = '%05d' % ix
            frame_n = config.test_video + "_" + num
            f_res = out_df[out_df["frame"].str.contains(frame_n)]
            bboxes = f_res.loc[:,["min_x","min_y","max_x","max_y"]].values.tolist()
            pre_labels = f_res.loc[:,"pred_action_label"].values.tolist()
            results.append([
                (
                    [
                        int(bboxes[i][0]*w),# int(bboxes[i][0]*w) - int(bboxes[i][2]*w/2), 
                        int(bboxes[i][1]*h),# int(bboxes[i][1]*h) - int(bboxes[i][3]*h/2), 
                        int(bboxes[i][2]*w),# int(bboxes[i][0]*w) + int(bboxes[i][2]*w/2), 
                        int(bboxes[i][3]*h)# int(bboxes[i][1]*h) + int(bboxes[i][3]*h/2)
                    ],
                    str(int(pre_labels[i])),
                    str(1),
                )
                for i in range(len(bboxes))
            ])
    vis_frames = visualize(frames, results)
    vid = mpy.ImageSequenceClip([x[:, :, ::-1].copy() for x in vis_frames],
                                    fps=config.output_fps)
    vid.write_videofile(os.path.join(config.out_path, "test_rf_result.mp4"))
    # tmp_frame_dir = osp.dirname(frames_path[0])
    # shutil.rmtree(tmp_frame_dir)

if __name__ == '__main__':
	main()

