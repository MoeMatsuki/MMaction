from mm2act import mm2Act
import conf_rf
import os

RFmodel = "/home/moe/MMaction/config/clf_model_WB.pkl"
out_csv = "rf_result.csv"
dir = "/home/moe/MMaction/result_slowonly_val"

predict_rf = mm2Act(conf_rf, RFmodel)
for curDir, dirs, files in os.walk(dir):
    if "test_mmaction.csv" in files:
        print(curDir)
        csv_path = os.path.join(curDir, "test_mmaction.csv")
        out_csv_path = os.path.join(curDir, out_csv)
        predict_rf(csv_path, out_csv_path)