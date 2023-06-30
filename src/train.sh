#!/bin/sh
# conf_path="config/prediction_slowonly.py"
conf_path="config/prediction_slowonly.py"
# video="KOKUYO_data/Webmtg_221226_01.MOV"

# python src/preprocessing/converter_json.py ${conf_path}
# python src/preprocessing/converter.py ${conf_path}
# python src/train_mmaction/train.py ${conf_path} --gpus 1 --validate
python src2/preprocess/converter.py
python src/train_mmaction/train.py ${conf_path}

# python src/train_mmaction/convert_forRF.py ${conf_path}
# python src/train_mmaction/train_rf.py ${conf_path} --RFmodel "clf_model_teambuild.pkl"