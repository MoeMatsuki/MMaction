import pickle
import mmcv
import numpy as np
import pandas as pd
from common import get_label

model_path = 'work_dirs_kokuyo2/clf_model.pkl'
conf_path = "config/prediction_slowonly.py"
config = mmcv.Config.fromfile(conf_path)

import matplotlib.pyplot as plt


with open(model_path, mode='rb') as f:  # with構文でファイルパスとバイナリ読み来みモードを設定
        model = pickle.load(f)
label_map = get_label(config)
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