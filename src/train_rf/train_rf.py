import setup as setup
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mmcv
import pickle
from common import get_label

conf_path = "config/default.py"
config = mmcv.Config.fromfile(conf_path)
dataset = pd.read_csv("/home/moe/MMaction/test_rf.csv", header=None)
model_path = 'work_dirs_kokuyo2/clf_model.pkl'

df = pd.read_csv("test_rf.csv")
# convert_str = lambda x: x.replace(" ", "_")
# df_action = df["action_label"].map(convert_str)
label_map = get_label(config)
labels = list(label_map.values())
df = df.dropna(how='any')
print(labels, df)

""" モデル学習 """
# 変数定義
X = df.loc[:, labels].values      # 説明変数
y = df["action_label"].values     # 目的変数（住宅価格の中央値）

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# ランダムフォレスト回帰
clf = RandomForestClassifier(max_depth=2, random_state=0)
# モデル学習
clf.fit(X_train, y_train)

# 推論
y_train_pred = clf.predict(X_train)
y_test_pred  = clf.predict(X_test)

with open(model_path, 'wb') as f:
	pickle.dump(clf, f)


# """ グラフ可視化 """
# # flatten：1次元の配列を返す、argsort：ソート後のインデックスを返す
# sort_idx = X_train.flatten().argsort()

# # 可視化用に加工
# X_train_plot  = X_train[sort_idx]
# Y_train_plot  = y_train[sort_idx]
# train_predict = forest.predict(X_train_plot)

# 予測値(Train）
y_train_pred = clf.predict(X_train)
# print(y_train_pred)

from sklearn.metrics import accuracy_score

""" 正解率 """
print(accuracy_score(y_train, y_train_pred))

# 可視化
# plt.scatter(X_train_plot, Y_train_plot, color='lightgray', s=70, label='Traning Data')
# plt.plot(X_train_plot, train_predict, color='blue', lw=2, label="Random Forest Regression")    

# # グラフの書式設定
# plt.xlabel('LSTAT（低所得者の割合）')
# plt.ylabel('MEDV（住宅価格の中央値）')
# plt.legend(loc='upper right')
# plt.show()

