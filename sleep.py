import pandas as pd
import os
import csv
import numpy as np

# read by default 1st sheet of an excel file




directory = r'C:\Users\madhu\Documents\NLSAP_Stress\data\sleep'
for filename in os.listdir(directory):
    path = os.path.join(directory, filename)
    df = pd.read_csv(path)
    min_win_RR = df["win_RR"].min()
    mean_win_RR = df["win_RR"].mean()
    max_win_RR = df['win_RR'].max()

    min_hf_lf = df["hf_lf"].min()
    mean_hf_lf = df["hf_lf"].mean()
    max_hf_lf = df['hf_lf'].max()

    min_hf = df["hf"].min()
    mean_hf = df["hf"].mean()
    max_hf = df['hf'].max()

    min_lf = df["lf"].min()
    mean_lf = df["lf"].mean()
    max_lf = df['lf'].max()

    min_mean_max_sleep_deep = np.array([filename, min_win_RR,mean_win_RR,max_win_RR, min_hf_lf,mean_hf_lf,max_hf_lf, min_hf,mean_hf,max_hf,min_lf,mean_lf,max_lf])
    with open(r'C:\Users\madhu\Documents\NLSAP_Stress\data\min_mean_max_sleep_deep', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(min_mean_max_sleep_deep)




#
# df = pd.read_excel('C:/Users/madhu/Documents/NLSAP_Stress/Hariprasad_output_json.xlsx', engine='openpyxl')
#
# df = df[df.sleepQuality == 3]
# print(df)
# require_cols = [4, 10]
#
#
# required_df = pd.read_excel(r'C:\Users\madhu\Downloads\Hariprasad_output_json (1).xlsx', usecols = require_cols)
