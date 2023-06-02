#!/bin/sh
# conda activate open-mmlab

conf_path="config/prediction_slowonly.py"
video="download/KOKUYO_data/IMG_1817.MOV"
base_name=`basename ${video} | sed 's/\.[^\.]*$//'`
echo ${base_name}
out_path="download/KOKUYO_data/result/${base_name}"

mkdir -p ${out_path}
mkdir -p tmp/${base_name}
ffmpeg -i ${video} -r 10 tmp/${base_name}/img_%06d.jpg

python src/prediction_mmaction/demo_spatiotmp_det3.py ${conf_path} --video ${video} --out ${out_path}
python src/predict_rf.py/predict_rf.py  ${conf_path} ${video} ${out_path} ${out_path}