conf.pyをいじる

# 学習と評価
> python src/randomforest/train_rf.py

※conf.pyのtrue_idsをいじると、train_rf.csvが変わる（目的変数が変わる）ことを確認してください。
※時系列処理に変更する場合は、convert_forRF.pyの中をいじるといいかもです。やりやすい方法でおまかせします。

# ラベルの分布を分析
> python src/randomforest/analysis.py

※"analysis.csv"と'pandas_analysis.png'が出力されます。

# 学習したモデルの重要度を分析
> python src/randomforest/analysis_rf.py

※"import_rf.csv"と'importance_rf.png'が出力されます。

# テストして結果を動画に出力
> python src/randomforest/predict_rf.py 

※conf.pyの下の方に定義しているファイルが処理されます。動画も出力されるので確認のときどうぞ。