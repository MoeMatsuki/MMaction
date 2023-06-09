import pandas as pd
from common import load_label_map
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    # currently only support plot curve and calculate average train time
    parser.add_argument('--conf_path', default=None)
    args = parser.parse_args()
    return args

args = parse_args()
if args.conf_path is None:
    import conf
    config = conf
else:
    from mmengine.config import Config
    config = Config.fromfile(args.conf_path)

# 学習データ
df = pd.read_csv(config.rf_train_csv)
df = df.dropna(how='any')
# ラベルデータ
label_map = load_label_map(config.label_map)
labels = list(label_map.values())

# 目的変数（住宅価格の中央値）
y = df["action_label"].values     
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

