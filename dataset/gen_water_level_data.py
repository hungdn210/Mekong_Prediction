import os
import pandas as pd

folder_path = "Filled_Gaps_Data/Water.Level"
all_dfs = []

# 遍历文件夹中的CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)

        # 读取CSV文件：第1列为时间，第3列为数据列
        df = pd.read_csv(file_path, usecols=[0, 2])

        # 设置列名：时间列 + 文件名（不含扩展名）
        col_name = os.path.splitext(filename)[0]
        df.columns = ["Timestamp (UTC+07:00)", col_name]
        df = df.set_index("Timestamp (UTC+07:00)")
        all_dfs.append(df)

merged_df = pd.concat(all_dfs, axis=1)

# 重置索引使时间列可见
merged_df = merged_df.reset_index()
merged_df.to_csv("Water_Level_Data.csv", index=False)
