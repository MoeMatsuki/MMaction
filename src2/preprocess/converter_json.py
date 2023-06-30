import json
import os
import cfg

def normalize(bbox, w, h):
    """bboxの正規化
    """
    return [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]

def convert(bbox):
    """bboxの正規化
    """
    w = (bbox[2] - bbox[0])/2
    h = (bbox[3] - bbox[1])/2
    box1 = bbox[0]+w
    box2 = bbox[1]+h
    box3 = bbox[2] - bbox[0]
    box4 = bbox[3] - bbox[1]
    
    return [box1, box2, box3, box4]

def fastlabel2txt(json_):
    """
    """
    w = json_["width"]
    h = json_["height"]
    annotations = json_["annotations"]
    result = []
    for human in annotations:
        attribute = human["attributes"]
        point = convert(human["points"])
        point = [str(i) for i in normalize(point, w, h)]
        # 人ラベル=0を書き込み
        line = ["0"]; line.extend(point)
        result.append(line)
        for a in attribute:
            value = a["value"]
            type = a["type"]
            if type == "checkbox":
                for activity_id in value:
                    # 行動クラスの書き込み
                    line = [activity_id]; line.extend(point)
                    result.append(line)
            if type == "select":
                # ラベルが存在しないものはスキップ
                if value == "":
                    continue
                # 行動クラスの書き込み
                line = [value]; line.extend(point)
                result.append(line)
    return result

def save_txt(txt_list, file_path):
    text_file = open(file_path, "wt")

    for t in txt_list:
        txt = " ".join(t)
        text_file.write(txt)
        text_file.write("\n")

    text_file.close()

def main():
    dir = cfg.JSON_DIR
    out_dir = cfg.TRAIN_TXT_DIR

    for label in os.listdir(dir):
        label_dir = os.path.join(dir, label)
        out_label = os.path.join(out_dir, label)
        if os.path.isdir(label_dir):
            for img_name in os.listdir(label_dir):
                img_dir = os.path.join(label_dir, img_name)
                out_img = os.path.join(out_label, img_name)
                os.makedirs(out_img, exist_ok=True)
                if os.path.isdir(img_dir):
                    # 動画名とフレーム数
                    print(img_name, len(os.listdir(img_dir)))
                    for img_file in os.listdir(img_dir):
                        if ".json" in img_file:
                            path = os.path.join(img_dir, img_file)
                            out_file = os.path.join(out_img, img_file.replace(".json", ".txt"))

                            # json読み取り
                            with open(path) as f:
                                json_ = json.load(f)
                            # json読み取り
                            txt_list = fastlabel2txt(json_)
                            save_txt(txt_list, out_file)

if __name__ == "__main__":
    main()
