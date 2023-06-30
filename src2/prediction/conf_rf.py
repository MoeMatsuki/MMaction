import os

anno_root = "fastlabel1/annotations"

# 上位行動
action_label = f"{anno_root}/classes_en2.txt"
# 下位行動
label_map = f"{anno_root}/classes_en.txt"

mm_csv = "/home/moe/MMaction/20230623_test_res/test_mmaction.csv"
mm2RF_csv = "/home/moe/MMaction/20230623_test_res/test_rf.csv"
mm_thresh = 1
flag_multi_cls = False
true_ids = [3,4]
exclude_sample_ids = []