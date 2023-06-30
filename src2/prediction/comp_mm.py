import pandas as pd
import os
from natsort import natsorted


target = "IMG_0568"
mm_res = f"/home/moe/MMaction/result_slowfast/{target}/test_mmaction.csv"
out_res = f"/home/moe/MMaction/result_slowfast/{target}/mm_result.csv"
img_path = f"tmp/{target}"

img_names = natsorted(os.listdir(img_path))
df = pd.read_csv(mm_res)

f = lambda x: x.split("/")[-1]
df["base_name"] = df["frame"].map(f)
df = df.set_index("base_name")
df_indexes = list(df.index)
col_names = df.columns

df_ix = 0
add_list = []
for d in img_names:
    if d in df_indexes:
        print(add_list)
        value = df.iloc[df_ix, :]
        v = dict(zip(col_names, value))
        add_df = pd.DataFrame(dict(zip(add_list, [v]*len(add_list))))
        df = pd.concat([df, add_df.T], axis=0)
        add_list = []
        # if (df_ix+1) < len(df):
        df_ix += 1
    else:
        add_list.append(d)
df = df.reindex(img_names)
df.to_csv(out_res)



