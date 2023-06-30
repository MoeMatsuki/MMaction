import os

anno_root = "fastlabel1/annotations"
# アノテーション済みファイル
anno_train_csv = os.path.join(anno_root, "train.csv")
anno_val_csv = os.path.join(anno_root, "val.csv")

# 学習データ
rf_train_csv = os.path.join(anno_root, "train_rf_new.csv")
# 評価用データ
rf_val_csv = os.path.join(anno_root, "val_rf_new.csv")
# モデルの名前
rf_model = "clf_model_WB.pkl"

# 結果を保存するディレクトリ
work_dir = 'work_dirs'
os.makedirs(work_dir, exist_ok=True)
rf_pickle_path = os.path.join(work_dir, rf_model)

# 識別対象とする上位アクティビティのid (リストにないものは除外)
#target_ids = [1, 2, 3, 4,5]
target_ids = [1, 2, 5]
# 多クラス分類をするか２値分類をするかどうか
flag_multi_cls = False
# ２値クラス分類のときのTrueとなるid を定義
# true_ids = [3, 4] # Team vs Solo
true_ids = [2, 5]
# このidを含むサンプルは除外される
exclude_sample_ids = [] #[1, 123]

# 上位行動
action_label = f"{anno_root}/classes_en2.txt"

# 下位行動
label_map = f"{anno_root}/classes_en.txt"

# 対象として扱う出現フレーム数の閾値
frm_thresh = 40

# testの設定ファイル
img_root = "fastlabel1/train_img"
test_video = "IMG_0567"
result_rf_path = "result_rf.csv"
out_path=f"fastlabel1/result/{test_video}"
output_fps = 5
