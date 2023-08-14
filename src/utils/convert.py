from converter import convert_img_filename
import glob
import os

img_dir_path = "/home/moe/MMaction/data/20230630_camera_test/original_img/8F_9"
out_put = "/home/moe/MMaction/data/20230630_camera_test/convert_img/8F_9"
os.makedirs(out_put, exist_ok=True)
img_files = glob.glob(os.path.join(img_dir_path, '*.jpg'))
convert_img_filename(img_files, out_put)
img_dir_path = "/home/moe/MMaction/data/20230630_camera_test/original_img/8F_10"
out_put = "/home/moe/MMaction/data/20230630_camera_test/convert_img/8F_10"
os.makedirs(out_put, exist_ok=True)
img_files = glob.glob(os.path.join(img_dir_path, '*.jpg'))
convert_img_filename(img_files, out_put)