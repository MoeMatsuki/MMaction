annotation_csv = "/home/moe/MMaction/fastlabel_test/annotations/train.csv"
train_dir = "/home/moe/MMaction/fastlabel1/train_img"

import pandas as pd

df = pd.read_csv(annotation_csv, header=None)
print(len(df))
c = df[0].unique()
c.sort()
with open('matsuki.txt', 'w') as f:
    for d in c:
        f.write("%s\n" % d)