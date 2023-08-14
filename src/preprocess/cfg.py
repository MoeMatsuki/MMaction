import os

JSON_DIR = "fastlabel1/jsons/"
TRAIN_TXT_DIR =  "fastlabel1/annotation/"

# ↓上２つを変更すれば良い
TRAIN_IMG_DIR = 'fastlabel1/train_img'
TRAIN_TXT_DIR = 'fastlabel_test/annotation'#converter_jsonの出力ディレクトリ
data_root = "fastlabel1/convert_img"
anno_root = "/home/moe/MMaction/fastlabel1/annotations"
anno_train_csv = os.path.join(anno_root, "20230620_train.csv")
rf_train_csv = os.path.join(anno_root, "20230620_train_rf.csv")

# ラベル
label_map = f"{anno_root}/classes_en.txt" 
action_label = f"{anno_root}/classes_en2.txt"
flag_multi_cls = True
true_ids = [3,4]
exclude_sample_ids = []#[1, 123]

# ↓このまま無視でよい
anno_val_csv = os.path.join(anno_root, "val.csv")
rf_val_csv = os.path.join(anno_root, "val_rf.csv")
val_videos = []#["IMG_1936_5", "IMG_1936_10", "IMG_1811", "IMG_0598", "IMG_0575"] #["IMG_1936_5", "IMG_1936_10", "IMG_1811", "IMG_0598", "IMG_0575"]
