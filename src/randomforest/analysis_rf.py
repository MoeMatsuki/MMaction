import pickle
import numpy as np
import pandas as pd
from common import load_label_map
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    parser.add_argument('--conf_path', default=None)
    parser.add_argument('--RFmodel', default=None)
    args = parser.parse_args()
    return args

args = parse_args()
if args.conf_path is None:
    import conf
    config = conf
else:
    from mmengine.config import Config
    config = Config.fromfile(args.conf_path)

if args.RFmodel is not None:
    config.rf_model =  args.RFmodel
model_path = config.rf_pickle_path



import matplotlib.pyplot as plt

with open(model_path, mode='rb') as f:  # with構文でファイルパスとバイナリ読み来みモードを設定
    model = pickle.load(f)
label_map = load_label_map(config.label_map)
labels = list(label_map.values())

fti = model.feature_importances_
indices = np.argsort(fti)[::-1]

print('Feature Importances:')
for i in indices:
    print('\t{0:20s} : {1:>.6f}'.format(labels[i], fti[i]))
label_sort = [labels[i] for i in indices]
print(len(fti[indices]),len(label_sort))
dic = dict({"label": label_sort, "importance":fti[indices]})
out_csv = pd.DataFrame(dic)
out_csv.to_csv("import_rf.csv")

plt.figure(figsize = (20,15))
plt.barh(y = range(len(fti)), width = fti[indices])
plt.yticks(ticks = range(len(label_sort)), labels = label_sort)
plt.savefig("importance_rf.png")   # プロットしたグラフをファイルsin.pngに保存する
plt.show()