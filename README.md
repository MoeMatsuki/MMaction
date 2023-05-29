# 環境
git clone https://github.com/MoeMatsuki/MMaction.git
cd MMaction
git clone https://github.com/open-mmlab/mmaction2.git

# 20230529

slowonlyの学習と推論をするデモ。[ここ](https://drive.google.com/drive/folders/1-E9wy4dasna7wYiLxlTGgyFAfehD82O8)からdownload.zipをダウンロードして展開したものをMMACTIONの下に置いてください。カレントディレクトリはMMACTIONで、以下のコマンドを順に実装してください。

conf_path="config/prediction_slowonly.py"
python src/preprocessing/converter.py ${conf_path}
python src/train_mmaction/train.py ${conf_path} --gpus 1 --validate


video="download/KOKUYO_data/IMG_1817.MOV"
base_name=`basename ${video} | sed 's/\.[^\.]*$//'`
out_path="download/KOKUYO_data/result/${base_name}"

mkdir ${out_path}
mkdir tmp/${base_name}
ffmpeg -i ${video} -r 10 tmp/${base_name}/img_%06d.jpg

python src/prediction_mmaction/demo_spatiotmp_det3.py ${conf_path} --video ${video} --out ${out_path}


## CPUの場合

※この方法では、実装にすごく時間がかかるのでオススメしない

1. 環境構築   
    env/README.mdの `CPU+docker` を確認する

2. demoの実装  
    - [VIDEO DEMO 参考](https://github.com/open-mmlab/mmaction2/tree/master/demo#video-demo)
        1. checkpointのダウンロード(初回だけ実装)
        ```
        mkdir checkpoints
        cd checkpoints
        wget https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth
        ```
        2. 実装するためのコマンド
        ```
        python demo.py mmaction2/configs/recognition/tsn/tsn_r50_inference_1x1x3_100e_kinetics400_rgb.py checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth mmaction2/demo/demo.mp4 mmaction2/tools/data/kinetics/label_map_k400.txt --use-frames --device cpu
        ```
        → 結果が `print`される (↓ 例)
        ```
        The top-5 labels with corresponding scores are:
        arm wrestling:  25.668127
        rock scissors paper:  9.766644
        shaking hands:  8.809238
        clapping:  8.333139
        stretching leg:  7.594356
        ```
        
        [demo.pyを実装するための手順メモ](memo/)





    - [SPATIO TEMP 参考](https://github.com/open-mmlab/mmaction2/tree/master/demo#spatiotemporal-action-detection-video-demo)

        1. checkpointのダウンロード（初回だけ実装）
        ```
        mkdir checkpoints
        cd checkpoints

        # slow-only model
        wget https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth

        # detection model
        wget http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth
        ```
        2. なんかエラーが起きたのでpip install
        ```
        pip install moviepy
        ```

        3. 実装するためのコマンド
        ```
        python demo_spatiotmp_det.py --video mmaction2/demo/demo.mp4 --config mmaction2/configs/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py --checkpoint checkpoints/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth --det-config mmaction2/demo/faster_rcnn_r50_fpn_2x_coco.py --det-checkpoint checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth  --det-score-thr 0.9  --action-score-thr 0.5  --label-map mmaction2/tools/data/ava/label_map.txt  --predict-stepsize 8  --output-stepsize 4 --output-fps 6 --device cpu --out-filename sp_demo_output.mp4
        ```
        → 結果の動画が出力される
    
## GPUの場合

1. 環境構築
    env/README.mdの `CPU+docker` を確認する


## memo
ffmpeg -i KOKUYO_data/convert/IMG_1819/image_%05d.jpg -vcodec libx264 -pix_fmt yuv420p out.mp4

python train.py slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py --gpus 1 --validate

python demo_spatiotmp_det.py --video KOKUYO_data/Webmtg_221226_01.MOV --config mmaction2/configs/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py --checkpoint work_dirs_kokuyo2/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/epoch_2.pth --det-config mmaction2/demo/faster_rcnn_r50_fpn_2x_coco.py --det-checkpoint checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth  --det-score-thr 0.9  --action-score-thr 0.5  --label-map KOKUYO_data/annotations/classes_en.txt  --predict-stepsize 8  --output-stepsize 4 --output-fps 6 --out-filename sp_demo_output_train.mp4

python demo_spatiotmp_det.py --video KOKUYO_data/Webmtg_221226_01.MOV --config config/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py --checkpoint work_dirs_kokuyo2/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/epoch_2.pth --det-config mmaction2/demo/faster_rcnn_r50_fpn_2x_coco.py --det-checkpoint checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth  --det-score-thr 0.9  --action-score-thr 0.5  --label-map KOKUYO_data/annotations/classes_en.txt  --predict-stepsize 8  --output-stepsize 4 --output-fps 6 --out-filename sp_demo_output_train.mp4