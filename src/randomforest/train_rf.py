import os
import pickle
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from common import load_label_map
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
from convert_forRF import ConvertRF
import collections

def parse_args():
    parser = argparse.ArgumentParser()
    # currently only support plot curve and calculate average train time
    parser.add_argument('--conf_path', default=None)
    parser.add_argument('--RFmodel', default=None)
    args = parser.parse_args()
    return args


class TrainRF:
	def __init__(self, config, rfmodel):
		# ランダムフォレスト用のcsvに変換
		self.converter = ConvertRF(config)
		self.converter(config.anno_train_csv, config.rf_train_csv)
		self.converter(config.anno_val_csv, config.rf_val_csv)

		# rfモデルを指定したい場合は変更
		if rfmodel is not None:
			config.rf_model =  rfmodel
		
		# チェック
		assert os.path.isfile(config.rf_train_csv), f"{config.rf_train_csv} is not exist"
		assert os.path.isfile(config.rf_val_csv), f"{config.rf_val_csv} is not exist"

		# 学習データ
		self.rf_val_csv = config.rf_val_csv
		self.rf_train_csv = config.rf_train_csv

		# 保存するモデルの名前
		self.model_path = config.rf_pickle_path

		# 下位行動ラベル
		self.l_action_label = load_label_map(config.label_map)
		# 上位行動ラベル
		action_label = load_label_map(config.action_label)
		h_action_label_ids = list(action_label.keys())

		# 下位行動が目的変数の場合、説明変数から下位行動を削除する
		if len(config.true_ids) != 0:
			for true_id in config.true_ids:
				if true_id in h_action_label_ids:
					continue
				del self.l_action_label[true_id]

	def print_result(self, y, y_pred):
		print(confusion_matrix(y, y_pred))
		print("正解率: " + str(accuracy_score(y, y_pred)))
		print("RECALL: " + str(recall_score(y, y_pred)))
		print("PRECISION: " + str(precision_score(y, y_pred)))
		print("")

	def eval(self):
		df_val = pd.read_csv(self.rf_val_csv)
		df_val = df_val.dropna(how='any')
		with open(self.model_path, 'rb') as f:
			clf = pickle.load(f)  # 復元
		X_val = df_val.loc[:, list(self.l_action_label.values())].values
		y_val = df_val["action_label"].values 

		y_val_pred  = clf.predict(X_val)
		print("validation prediction result")
		self.print_result(y_val, y_val_pred)


	def __call__(self):

		# 目的変数がNoneのものは対象外
		df_train = pd.read_csv(self.rf_train_csv)
		df_train = df_train.dropna(how='any')
		
		""" モデル学習 """
		# 変数定義
		X_train = df_train.loc[:, list(self.l_action_label.values())].values # 説明変数
		y_train = df_train["action_label"].values     # 目的変数

		# ランダムフォレスト
		clf = RandomForestClassifier(max_depth=2, random_state=0)
		# モデル学習
		clf.fit(X_train, y_train)

		# 推論
		y_train_pred = clf.predict(X_train)
		""" 正解率 """
		print("train prediction result")
		self.print_result(y_train, y_train_pred)
		
		with open(self.model_path, 'wb') as f:
			pickle.dump(clf, f)

		self.eval()

def main():
	args = parse_args()
	if args.conf_path is None:
		import conf
		config = conf
	else:
		from mmengine.config import Config
		config = Config.fromfile(args.config)
	train_rf = TrainRF(config, args.RFmodel)
	train_rf()

if __name__ == '__main__':
	main()

