import cv2
import skimage.util
import numpy as np
from pathlib import Path

def scanning(img):
    h, w = img.shape[:2]    # グレースケール画像のサイズ取得（カラーは3）
    x = int(w/2)           # 領域の横幅
    y = int(h/2)           # 領域の高さ
    x_step = int(x*0.8)             # 領域の横方向へのずらし幅
    y_step = int(y*0.8)            # 領域の縦方向へのずらし幅
    x0 = 0                  # 領域の初期値x成分
    y0 = 0                  # 領域の初期値y成分
    j = 0                   # 縦方向のループ指標を初期化
    imgs = []
    # 縦方向の走査を行うループ
    while y + (j * y_step) <= h:
        i = 0                   # 横方向の走査が終わる度にiを初期化
        ys = y0 + (j * y_step)  # 高さ方向の始点位置を更新
        yf = y + (j * y_step)   # 高さ方向の終点位置を更新
 
        # 横方向の走査をするループ
        while x + (i * x_step) <= w:
            roi = img[ys:yf, x0 + (i * x_step):x + (i * x_step)]    # 元画像から領域をroiで抽出
            imgs.append(roi)
 
            i = i + 1   # whileループの条件がFalse（横方向の端になる）まで、iを増分
        j = j + 1       # whileループの条件がFalse（縦方向の端になる）まで、jを増分
    return imgs

def split(img):
    rows = 2  # 行数
    cols = 2  # 列数

    chunks = []
    for row_img in np.array_split(img, rows, axis=0):
        for chunk in np.array_split(row_img, cols, axis=1):
            chunks.append(chunk)
    print(chunks)
    return chunks

def split_patch(img):
    print(img.shape)
    split = skimage.util.view_as_blocks(img, (2, 3, 2))
    print(split)

def out(chunks):
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    for i, chunk in enumerate(chunks):
        save_path = output_dir / f"chunk_{i:02d}.png"
        cv2.imwrite(str(save_path), chunk)

img_path = "/home/moe/MMaction/tmp/2022-05-11_15-00-00/img_000001.jpg"
img = cv2.imread(img_path)
chunks = scanning(img)
print(len(chunks))
out(chunks)