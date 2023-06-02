#!/bin/sh
conf_path="config/prediction_slowonly.py"
# video="KOKUYO_data/Webmtg_221226_01.MOV"

# python src/preprocessing/converter.py ${conf_path}
python src/preprocessing/converter_json.py ${conf_path}
python src/train_mmaction/train.py ${conf_path} --gpus 1 --validate

python src/train_rf/convert_forRF.py ${conf_path} "clf_model_teambuild.pkl"
python src/train_rf/train_rf.py ${conf_path}