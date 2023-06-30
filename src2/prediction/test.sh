#!/bin/sh
# conda activate open-mmlab

# preprocess
# python src2/preprocess/converter.py
# python src2/preprocess/convert_forRF.py

# conf_path="config/prediction_slowonly.py"
conf_path="config/prediction_slowfast.py"
video="IMG_0567"
base_name=`basename ${video} | sed 's/\.[^\.]*$//'`
# echo ${base_name}
out_path="result_slowfast/${base_name}"
# or 
outdir="result_slowfast_val"

# mkdir -p ${out_path}
mkdir -p ${outdir}
# mkdir -p tmp/${base_name}
# ffmpeg -i ${video} -r 1 tmp/${base_name}/img_%06d.jpg

# python src2/prediction/prediction_spatiotmp.py ${conf_path} --video ${video} --out ${out_path}
python src2/prediction/prediction_model.py ${conf_path} --outdir ${outdir}
# echo ${out_path}
# python src2/prediction/eval.py --out ${out_path}
python src2/prediction/eval.py --outdir ${outdir}

# python src/predict_rf.py/predict_rf.py  ${conf_path} ${video} ${out_path} ${out_path}