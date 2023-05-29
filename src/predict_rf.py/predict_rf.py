import pickle
import pandas as pd
import mmcv
import cv2
import os
import argparse
import os.path as osp
import shutil
from train_rf.common import get_label, dense_timestamps, convert_result, visualize



test_df = "test_mmaction.csv"
output_path = "test_rf.csv"

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    parser.add_argument('conf_path')
    parser.add_argument('video')
    parser.add_argument('input')
    parser.add_argument('out')
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
    conf_path = args.conf_path
    config = mmcv.Config.fromfile(conf_path)

    if args.model is None:
        model_path = conf_path.rf_model
    else:
        model_path = args.model
    # frames_path, _ = frame_extraction(video)
    with open(model_path, mode='rb') as f:  # with構文でファイルパスとバイナリ読み来みモードを設定
        model = pickle.load(f)

    label_map = get_label(config)
    labels = list(label_map.values())
    print(os.path.join(args.input, test_df))
    df = pd.read_csv(os.path.join(args.input, test_df))
    # df = df.dropna(how='any')

    X = df.loc[:, labels].values      # 説明変数

    y_train_pred = model.predict(X)
    out_df = pd.concat([df, pd.Series(y_train_pred)], axis=1)
    out_df = out_df.rename(columns={0: "action_label"})
    out_df.to_csv(os.path.join(args.out, output_path), index = False)

    try:
        import moviepy.editor as mpy
    except ImportError:
        raise ImportError('Please install moviepy to enable output file')


    frames_path = out_df.loc[:, "frame"].to_list()
    frames_path = list(dict.fromkeys(frames_path))
    frames = [cv2.imread(f) for f in frames_path]

    results = convert_result(out_df, labels)
    vis_frames = visualize(frames, results)
    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames],
                                    fps=config["output_fps"])
    vid.write_videofile(os.path.join(args.out, "test_rf_result.mp4"))
    # tmp_frame_dir = osp.dirname(frames_path[0])
    # shutil.rmtree(tmp_frame_dir)

if __name__ == '__main__':
	main()

