import argparse
# import setup as setup
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mmcv
import pickle
from common import get_label, load_label_map
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from convert_forRF import ConvertRF

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    parser.add_argument('conf_path')
    args = parser.parse_args()
    return args


class TrainRF:
	def __init__(self, config):
		converter = ConvertRF(config)
		converter()
		assert os.path.isfile(config.rf_train_csv), f"{config.rf_train_csv} is not exist"
		assert os.path.isfile(config.rf_val_csv), f"{config.rf_val_csv} is not exist"
		self.config = config
		self.converter = ConvertRF(config)
		self.model_path = config.rf_pickle_path
		self.label_map = get_label(config)
		# self.labels = list(label_map.values())
		true_ids = config.true_ids
		action_label = load_label_map(config.action_label)
		action_label_ids = list(action_label.keys())

		# label_ids = list(label_map.keys())
		# if len(true_ids) != 0:
		# 	for true_id in true_ids:
		# 		if true_id in action_label_ids:
		# 			continue
		# 		print(label_ids.index(true_id))
		# 		self.labels.pop(label_ids.index(true_id))
		label_ids = list(self.label_map.keys())
		if len(true_ids) != 0:
			for true_id in true_ids:
				if true_id in action_label_ids:
					continue
				print("exclude colmns")
				print(self.label_map[true_id])
				del self.label_map[true_id]

	def eval(self):
		df_val = pd.read_csv(self.config.rf_val_csv)
		df_val = df_val.dropna(how='any')
		with open(self.model_path, 'rb') as f:
			clf = pickle.load(f)  # 復元
		X_val = df_val.loc[:, list(self.label_map.values())].values      # 説明変数
		y_val = df_val["action_label"].values 

		y_val_pred  = clf.predict(X_val)
		print("validation prediction result")
		print(confusion_matrix(y_val, y_val_pred))
		print(accuracy_score(y_val, y_val_pred))

	def __call__(self):
		config = self.config
		self.converter()
		df_train = pd.read_csv(config.rf_train_csv)
		
		# convert_str = lambda x: x.replace(" ", "_")
		# df_action = df["action_label"].map(convert_str)
		df_train = df_train.dropna(how='any')
		
		""" モデル学習 """
		# 変数定義
		X_train = df_train.loc[:, list(self.label_map.values())].values # 説明変数
		y_train = df_train["action_label"].values     # 目的変数

		# ランダムフォレスト回帰
		clf = RandomForestClassifier(max_depth=2, random_state=0)
		# モデル学習
		clf.fit(X_train, y_train)

		# 推論
		y_train_pred = clf.predict(X_train)

		with open(self.model_path, 'wb') as f:
			pickle.dump(clf, f)

		""" 正解率 """
		print("train prediction result")
		print(confusion_matrix(y_train, y_train_pred))
		print(accuracy_score(y_train, y_train_pred))
		self.eval()


# """ グラフ可視化 """
# # flatten：1次元の配列を返す、argsort：ソート後のインデックスを返す
# sort_idx = X_train.flatten().argsort()

# # 可視化用に加工
# X_train_plot  = X_train[sort_idx]
# Y_train_plot  = y_train[sort_idx]
# train_predict = forest.predict(X_train_plot)

# 予測値(Train）
# print(y_train_pred)


# 可視化
# plt.scatter(X_train_plot, Y_train_plot, color='lightgray', s=70, label='Traning Data')
# plt.plot(X_train_plot, train_predict, color='blue', lw=2, label="Random Forest Regression")    

# # グラフの書式設定
# plt.xlabel('LSTAT（低所得者の割合）')
# plt.ylabel('MEDV（住宅価格の中央値）')
# plt.legend(loc='upper right')
# plt.show()
def main():
	args = parse_args()
	conf_path = args.conf_path
	config = mmcv.Config.fromfile(conf_path)
	train_rf = TrainRF(config)
	train_rf()

if __name__ == '__main__':
	main()

