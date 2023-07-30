

# def frame_extraction(video_path: str,
#                   short_side: Optional[int] = None,
#                   out_dir: str = 'tmp'):
#     """Extract frames given video_path.

#     Args:
#         video_path (str): The video path.
#         short_side (int): Target short-side of the output image.
#             Defaults to None, means keeping original shape.
#         out_dir (str): The output directory. Defaults to ``'./tmp'``.
#     """
#     # Load the video, extract frames into OUT_DIR/video_name
#     target_dir = osp.join(out_dir, osp.basename(osp.splitext(video_path)[0]))
#     os.makedirs(target_dir, exist_ok=True)
#     # Should be able to handle videos up to several hours
#     # frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
#     frame_paths = os.listdir(target_dir)
    
#     # assert osp.exists(video_path), f'file not exit {video_path}'
#     # vid = cv2.VideoCapture(video_path)
#     # print(vid.set(cv2.CAP_PROP_FPS, 1))
#     # fps_setting = vid.get(cv2.CAP_PROP_FPS)
#     # print("FPS(Setting):",'{:11.02f}'.format(fps_setting))
#     # frame_paths = glob.glob(os.path.join(target_dir, '*.jpg'))
#     frame_paths = natsorted(frame_paths)
#     frame_paths = [os.path.join(target_dir, i) for i in frame_paths]
#     frames = [cv2.imread(i) for i in frame_paths]

#     # flag, frame = vid.read()

#     # frames = []
#     # frame_paths = []
#     cnt = 0
#     new_h, new_w = None, None
#     # while flag:
#     for ix, frame in enumerate(frames):
#         if short_side is not None:
#             if new_h is None:
#                 h, w, _ = frame.shape
#                 new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))
#             frame = mmcv.imresize(frame, (new_w, new_h))

#         # frames.append(frame)
#         # frame_path = frame_tmpl.format(cnt + 1)
#         # frame_paths.append(frame_path)
#         # frame_path = frame_paths[ix].format(cnt + 1)

#         # cv2.imwrite(frame_path, frame)
#         # cnt += 1
#         # flag, frame = vid.read()

#     return frame_paths, frames

import cv2
import glob

def scanning(img):
    h, w = img.shape[:2]    # グレースケール画像のサイズ取得（カラーは3）
    x = int(w/2 * 1.2)          # 領域の横幅
    y = int(h/2 * 1.2)        # 領域の高さ
    x_step = int(w/2*0.8)             # 領域の横方向へのずらし幅
    y_step = int(h/2*0.8)            # 領域の縦方向へのずらし幅
    x0 = 0                  # 領域の初期値x成分
    y0 = 0                  # 領域の初期値y成分
    j = 0                   # 縦方向のループ指標を初期化
    imgs = []
    # 縦方向の走査を行うループ
    while y + (j * y_step) <= h:
        i = 0                   # 横方向の走査が終わる度にiを初期化
        ys = y0 + (j * y_step)  # 高さ方向の始点位置を更新
        yf = y + (j * y_step)   # 高さ方向の終点位置を更新
        print(yf, h)
 
        # 横方向の走査をするループ
        while x + (i * x_step) <= w:
            roi = img[ys:yf, x0 + (i * x_step):x + (i * x_step)]    # 元画像から領域をroiで抽出
            imgs.append(roi)
 
            i = i + 1   # whileループの条件がFalse（横方向の端になる）まで、iを増分
        j = j + 1       # whileループの条件がFalse（縦方向の端になる）まで、jを増分
    return imgs
import os

def main():
    dir = "/home/moe/MMaction/2022年8月｜6F改修前/convert_img/area3_220823_IMG_0006"
    out_dir = "/home/moe/MMaction/2022年8月｜6F改修前/split_img"
    img_paths = glob.glob(f'{dir}/*.jpg')

    for img_path in img_paths:
        vid_name = img_path.split("/")[-2]
        img_name = img_path.split("/")[-1]
        img = cv2.imread(img_path)
        imgs = scanning(img)
        for ix, im in enumerate(imgs):
            id = str(ix)
            o_dir = os.path.join(out_dir, vid_name + f"_{id}")
            os.makedirs(o_dir, exist_ok=True)
            cv2.imwrite(os.path.join(o_dir, img_name), im)

if __name__ == "__main__":
    main()
