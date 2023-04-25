import pickle
import pandas as pd
import mmcv
import cv2
import os
from common import get_label, dense_timestamps, convert_result, visualize
conf_path = "config/default.py"
config = mmcv.Config.fromfile(conf_path)

model_path = 'work_dirs_kokuyo2/clf_model.pkl'
test_df = "/home/moe/MMaction/test_mmaction2.csv"
output_path = "/home/moe/MMaction/test_rf.csv"

with open(model_path, mode='rb') as f:  # with構文でファイルパスとバイナリ読み来みモードを設定
    model = pickle.load(f)

label_map = get_label(config)
labels = list(label_map.values())
df = pd.read_csv(test_df)
df = df.dropna(how='any')

X = df.loc[:, labels].values      # 説明変数

y_train_pred = model.predict(X)
out_df = pd.concat([df, pd.Series(y_train_pred)], axis=1)
out_df = out_df.rename(columns={0: "action_label"})


try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')


imgs = out_df.loc[:, "frame"].to_list()
frames_path = [
            "/".join(i.split("/")[1:])
            for i in imgs
        ]
frames_path = list(dict.fromkeys(frames_path))
print(frames_path)
frames = [cv2.imread(f) for f in frames_path]

results = convert_result(out_df, labels)
vis_frames = visualize(frames, results)
vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames],
                                fps=config["output_fps"])
vid.write_videofile("test_rf_result.mp4")