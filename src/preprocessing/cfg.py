import os

# ↓上２つを変更すれば良い
TRAIN_IMG_DIR = 'fastlabel1/train_img'
TRAIN_TXT_DIR = 'fastlabel1/annotation'#converter_jsonの出力ディレクトリ
data_root = "fastlabel1/convert_img"
anno_root = "fastlabel1/annotations"
anno_train_csv = os.path.join(anno_root, "train.csv")

# ↓このまま無視でよい
anno_val_csv = os.path.join(anno_root, "val.csv")
val_videos = [] #["IMG_1936_5", "IMG_1936_10", "IMG_1811", "IMG_0598", "IMG_0575"]
