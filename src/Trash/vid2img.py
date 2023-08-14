import glob
from os import makedirs
import cv2

from os.path import splitext, dirname, basename, join

def save_frames(video_path: str, frame_dir: str, 
                name="img", ext="jpg", max_frame_num=float("inf")):
    cap = cv2.VideoCapture(video_path)
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if not cap.isOpened():
        print("cannot open")
        return
    v_name = splitext(basename(video_path))[0]
    if frame_dir[-1:] == "\\" or frame_dir[-1:] == "/":
        frame_dir = dirname(frame_dir)
    frame_dir_ = join(frame_dir, v_name)
    print(f"{frame_dir_} is saved directory")

    makedirs(frame_dir_, exist_ok=True)
    base_path = join(frame_dir_, name)

    idx = 0
    count = 0
    while cap.isOpened():
        idx += 1
        ret, frame = cap.read()
        if ret and count < max_frame_num:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == 1:  # 0秒のフレームを保存
                cv2.imwrite("{}_{}.{}".format(base_path, "00000", ext),
                            frame)
            elif idx < cap.get(cv2.CAP_PROP_FPS):
                continue
            else:  # 1秒ずつフレームを保存
                second = int(cap.get(cv2.CAP_PROP_POS_FRAMES)/idx)
                filled_second = str(second).zfill(5)
                cv2.imwrite("{}_{}.{}".format(base_path, filled_second, ext),
                            frame)
                idx = 0
            count += 1
        else:
            break

def main():

    dir = "/home/moe/MMaction/data/230623_実験前テスト/original"
    save_dir = "/home/moe/MMaction/data/230623_実験前テスト/convert_img"

    vids = glob.glob(f'{dir}/*.mp4')
    for vid in vids:
        save_frames(vid, save_dir)
        print(f"{vid} is done")

if __name__ == "__main__":
    main()