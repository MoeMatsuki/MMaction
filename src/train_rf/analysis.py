import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mmcv
import pickle
from train_rf.common import get_label
import matplotlib as mpl
import matplotlib.pyplot as plt

conf_path = "config/default.py"
config = mmcv.Config.fromfile(conf_path)
model_path = 'work_dirs_kokuyo2/clf_model.pkl'

df = pd.read_csv("test_rf.csv")
# convert_str = lambda x: x.replace(" ", "_")
# df_action = df["action_label"].map(convert_str)
label_map = get_label(config)
labels = list(label_map.values())
df = df.dropna(how='any')

""" モデル学習 """
# 変数定義
# X = df.loc[:, labels].values      # 説明変数
y = df["action_label"].values     # 目的変数（住宅価格の中央値）

df_analysis = pd.DataFrame()
for act in list(set(y)):
	df_filter = df[df["action_label"] == act]
	df_act = df_filter.loc[:, labels].sum(axis=0)
	df_act = df_act.rename(act)
	df_act = df_act / sum(df_act)
	df_analysis = pd.concat([df_analysis, df_act], axis=1)
df_analysis.to_csv(f"analysis.csv")

ax = df_analysis.plot.bar(figsize=(50,10), fontsize=10)
ax.figure.savefig('pandas_analysis.png')

