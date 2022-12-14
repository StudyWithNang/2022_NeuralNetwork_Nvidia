import pandas as pd

filepath = "all_data.csv"

df_data = pd.read_csv(filepath)

df_file1 = df_data.sample(frac=0.8, random_state=2022).sort_index()
df_file2 = df_data.drop(df_file1.index)

df_file1.to_csv('train1.csv', sep=',', na_rep='NaN', index=False)  # 혹시 몰라서 파일 이름을 train1로
df_file2.to_csv('test.csv', sep=',',na_rep='NaN', index = False)

##

filepath2 = "train1.csv"

df_data = pd.read_csv(filepath2)

df_file1 = df_data.sample(frac=0.8, random_state=2022).sort_index()
df_file2 = df_data.drop(df_file1.index)

df_file1.to_csv('train.csv', sep=',', na_rep='NaN', index=False)  # 혹시 몰라서 파일 이름을 train1로
df_file2.to_csv('val.csv', sep=',',na_rep='NaN', index = False)
