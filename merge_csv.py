import glob
import pandas as pd
csv_files = glob.glob('dataset/*.csv')
print("찾은 CSV 파일들:", csv_files)

df_list = []
for file in csv_files:
    df = pd.read_csv(file)
    df_list.append(df)

merged_df = pd.concat(df_list, ignore_index=True)
merged_df = merged_df.sort_values('open_time')
merged_df.to_csv('merged_dataset/merged_dataset.csv', index=False)
print("Merged DataFrame:")
print(merged_df.head())
