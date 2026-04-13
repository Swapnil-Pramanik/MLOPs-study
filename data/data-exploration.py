import pandas as pd 

df_2019 = pd.read_csv("MLOPs-study/data/yellow_tripdata_2019-01.csv")
df_2020 = pd.read_csv("MLOPs-study/data/yellow_tripdata_2020-01.csv")

print(df_2019.columns.tolist())
print(df_2020.columns.tolist())