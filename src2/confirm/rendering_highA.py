import cv2
import copy
import numpy as np

def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))

def load_label_map(file_path):
    """Load Label Map.

    Args:
        file_path (str): The file path of label map.
    Returns:
        dict: The label map (int -> label name).
    """
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}

plate_blue = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
plate_blue = plate_blue.split('-')
plate_blue = [hex2color(h) for h in plate_blue]
plate_green = '004b23-006400-007200-008000-38b000-70e000'
plate_green = plate_green.split('-')
plate_green = [hex2color(h) for h in plate_green]

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
THICKNESS = 1
FONTCOLOR = (255, 255, 255)  # BGR, white
LINETYPE = 1

def rendering_bbox(img, box, col=plate_blue[0]):
    """
    Args:
        img(np.array)
        box(np.array)
    """
    st, ed = tuple(box[:2]), tuple(box[2:])
    cv2.rectangle(img, st, ed, col, 2)
    return img

def rendering_text(img, box, label, n=0, cols=plate_blue,):
    st, ed = tuple(box[:2]), tuple(box[2:])
    location = (0 + st[0], 18 + n * 18 + st[1])
    textsize = cv2.getTextSize(label, FONTFACE, FONTSCALE,
                                THICKNESS)[0]
    textwidth = textsize[0]
    diag0 = (location[0] + textwidth, location[1] - 14)
    diag1 = (location[0], location[1] + 2)
    cv2.rectangle(img, diag0, diag1, cols[n + 1], -1)
    cv2.putText(img, label, location, FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)
    return img

def rendering_bboxes(img, boxes, labels, cols=plate_blue, f_scaring=True):
    """
    Args:
        img(np.array)
        box(np.array)
    """
    img_ = copy.deepcopy(img)
    num = len(plate_blue)
    for ix, box in enumerate(boxes):
        if f_scaring:
            # bbox
            box = scaring(img_, box)
        ixx = ix % num
        img_ = rendering_bbox(img_, box, plate_blue[ixx])
        img_ = rendering_text(img_, box, labels[ix])
    return img_

def scaring(img, box):
    h, w, _ = img.shape[:3]
    scale_ratio = np.array([w, h, w, h])
    return (box * scale_ratio).astype(np.int64)

test_video_dir = "/home/moe/MMaction/fastlabel1/train_img/"
train_csv = "/home/moe/MMaction/fastlabel_test/annotations/20230620_train.csv"
label_path = "/home/moe/MMaction/download/KOKUYO_data/annotations/classes_en3.txt"
out_dir = "out"

import os
import pandas as pd
from natsort import natsorted

flag_continue = True
df = pd.read_csv(train_csv, header=None)
img_names = df[0].unique()
for img_name in img_names:
    if not os.path.exists(os.path.join(test_video_dir, img_name)):
        print(f"{img_name} is not exist")
        flag_continue = False

assert flag_continue, "download images"

for img_name in img_names:
    print(img_name)
    test_video = os.path.join(test_video_dir, img_name)
    img_names = natsorted(os.listdir(test_video))
    f = lambda x: "_" + str(x * 30).zfill(4)
    df["frame"] = df[1].map(f)
    df["img_name"] = df[0] + df["frame"]
    label_map = load_label_map(label_path)

    target_box = {}
    for iname in img_names:
        img_name = iname.split(".")[0]
        d = df[df["img_name"] == img_name]

        d_filter = d[d.loc[:, 6] < 6]
        d_filter = d_filter[d_filter.loc[:, 6] > 0]
        box = d_filter.loc[:, [2,3,4,5]].values
        labels = [label_map[i] for i in list(d_filter.loc[:, 6].values)]
        target_box.update({iname:{"bbox": box, "labels":labels}})

    base = test_video.split("/")[-1]
    out_dir = os.path.join(out_dir, base)
    os.makedirs(out_dir, exist_ok=True)
    for img, value in target_box.items():
        test_img = os.path.join(test_video, img)
        out_img = os.path.join(out_dir, img)
        test_box = value["bbox"]
        labels = value["labels"]
        test_img = cv2.imread(test_img)
        res = rendering_bboxes(test_img, test_box, labels)
        cv2.imwrite(out_img, res)