import pandas as pd
import csv
import numpy as np
import json
import csv
from datetime import timedelta
import numpy as np
import json
import pandas as pd
import random

from movesense_read import Movesense

import os


# name_file = []
# date = []
# list_of_name = []
# list_n = []
# data_value = np.empty(shape=[0, 5])
#
#
# file_name = r'C:\Users\madhu\Documents\NLSAP_Stress\data\threshold_bounds'
# df = pd.read_csv(file_name)
# print(df)
# # mean_no = df["no"].mean()
# mean_low = df["low"].mean()
# min_low = df["low"].min()
# max_low = df['low'].max()
# mean_medium = df["medium"].mean()
# min_medium = df["medium"].min()
# max_medium = df['medium'].max()
# mean_high = df["high"].mean()
# min_high = df["high"].min()
# max_high = df['high'].max()
# # mean_veryhigh = df["veryhigh"].mean()
# stats = np.array([])
# stats = np.array(["Subject", min_low,mean_low,max_low,min_medium,mean_medium, max_medium, min_high, mean_high, max_high])
#
# import csv
#
# with open(r'C:\Users\madhu\Documents\NLSAP_Stress\data\statistics', 'a') as f:
#     writer = csv.writer(f)
#     writer.writerow(stats)

import pandas as pd
data= pd.read_json(r'C:\Users\madhu\Documents\NLSAP_Stress\data\Hariprasad_output_json.json')#the absolute path in explore
print(data)
#
# df_1 = pd.read_json('output_json.json', orient=str)
