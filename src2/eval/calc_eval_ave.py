import os
import pandas as pd

dir = "/home/moe/MMaction/result_slowfast_val"
all_df = None
count = 0
for curDir, dirs, files in os.walk(dir):
    if "low_activity.csv" in files:
        csv_path = os.path.join(curDir, "low_activity.csv")
        df = pd.read_csv(csv_path, index_col=0)
        if all_df is None:
            all_df = df
        else:
            all_df = all_df + df
        count += 1

print(all_df, count)
report_df_ave = all_df.iloc[:, :3] / int(count)
report_df_ave = pd.concat([report_df_ave, all_df.iloc[:, 3:]], axis = 1)
# report_df = pd.DataFrame(ave_report).T
report_df_ave.to_csv(os.path.join(dir, "ave_eval_low_activity.csv"))