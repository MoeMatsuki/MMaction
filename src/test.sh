#!/bin/sh
# conda activate open-mmlab

#conf_path="config/prediction_slowonly.py"
conf_path="config/prediction_slowfast.py"
video="IMG_0568"#"/home/moe/MMaction/tmp_input/品川南館08F_フロアD_2022-05-27/2022-05-27_16-00-00.mp4"
base_name=`basename ${video} | sed 's/\.[^\.]*$//'`
echo ${base_name}
out_path="result/${base_name}"

mkdir -p ${out_path}
# mkdir -p tmp/${base_name}
# ffmpeg -i ${video} -r 1 tmp/${base_name}/img_%06d.jpg

python src/prediction_mmaction/demo_spatiotmp_det3.py ${conf_path} --video ${video} --out ${out_path}
python src2/prediction/eval.py 

# python src/predict_rf.py/predict_rf.py  ${conf_path} ${video} ${out_path} ${out_path}