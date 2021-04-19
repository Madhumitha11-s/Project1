import numpy as np
import json
import csv
import pandas as pd
from numpy.random import rand
from numpy.random import seed
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import os
import seaborn as sb
import seaborn as sns
#from movesense_report import NLSAP_path

# path_for_threshold_csv = r'C:\Users\madhu\Documents\NLSAP_Stress\threshold_data\stress_thresholds_data.csv'
NLSAP_path = r'C:\Users\madhu\Documents\NLSAP_Stress'

#
# directory = os.path.join(NLSAP_path, "dumps", "") #navigate to ble dump
# for filename in os.listdir(directory):
#     path = os.path.join(directory, filename)
#     list_of_name = filename.split("_")
#     list_n = list_of_name[4].split(".")
#     subject_name = list_of_name[1]
#     name_file.append(list_of_name[1])
#     date = list_of_name[2]+"_"+list_of_name[3]+"_"+list_n[0]
#     file = open(path)
#     input_data_frame = json.load(file)
#
#     or_path = os.path.join(NLSAP_path, "data", "")
#     path = or_path + list_of_name[1] + '\\'

directory = os.path.join(NLSAP_path, "movesene_report_json_csv", "") #navigate to ble dump
for filename in os.listdir(directory):
    path = os.path.join(directory, filename)
    list_of_name = filename.split("_")
    subject_name = list_of_name[0]
    date = list_of_name[1]+"_"+list_of_name[2]+"_"+list_of_name[3]
    df = pd.read_csv(path)

    or_path_s = os.path.join(NLSAP_path, "threshold_data", "")
    temp_filename_s = str(subject_name) + "_stress_thresholds_data.json"
    path_s = or_path_s + temp_filename_s
    df_s = pd.read_json(path_s)
    print(df_s)


    clean_df = df[df.stressScores != 5.0]
    clean_df = clean_df[clean_df.stressScores != 4.0]
    df = clean_df

    threshold_bounds = np.array([])
    stress_levels = np.array([])
    stress_regions = np.array([])

    #for rr and hflf
    def get_stressscore(stress_score, no, low, med):
        stress_regions = np.array([])
        for element in stress_score:
            if element >= no :
                stress_regions = np.append(stress_regions, 0)
            elif element < no and element >= low :
                stress_regions = np.append(stress_regions, 1)
            elif element < low and element >= med :
                stress_regions = np.append(stress_regions, 2)
            elif element < med :
                stress_regions = np.append(stress_regions, 3)


        stress_levels = stress_regions
        return stress_levels

    stress_score_1 = np.asarray(df.win_RR)
    stress_score_2 = np.asarray(df.hf_lf)
    stress_levels_1 = get_stressscore(stress_score_1, int(df_s["win_RR_1"]),0.5*(int(df_s["win_RR_1"])+int(df_s["win_RR_2"])),int(df_s["win_RR_2"]))
    df['stress_levels_1'] = stress_levels_1
    stress_levels_2 = get_stressscore(stress_score_2, float(df_s["hf_lf_1"]),0.75*float(df_s["hf_lf_1"]),0.5*float(df_s["hf_lf_1"]))
    df['stress_levels_2'] = stress_levels_2
    # based on HF Lf
    stress_levels = (1.75*(stress_levels_1) + 1.75*(stress_levels_2))

    norm_stress_levels = stress_levels/np.max(stress_levels)
    df['new_stress_score'] = norm_stress_levels


    or_path_n = os.path.join(NLSAP_path, "updated_movesense_report", "")
    temp_filename_n = str(subject_name) +"_"+ str(date) + "_new_stress_thresholds_data.csv"
    path_n = or_path_n + temp_filename_n

    df.to_csv (path_n)
    matrix = np.corrcoef([stress_levels,df.stressScores,df.hf_lf,df.lf])
    x_axis_labels = ["new stress_s", "old stress_s", "hf_lf", "lf"] # labels for x-axis
    y_axis_labels = ["new stress_s", "old stress_s", "hf_lf", "lf"] # labels for y-axis

    sb.heatmap(matrix,  xticklabels=x_axis_labels, yticklabels=y_axis_labels,cmap="Blues", annot=True)
    plt.savefig(os.path.join(NLSAP_path, "plots", "coef","") + str(subject_name) +"_"+str(date)+ "_corrcoef.png")
    plt.clf()
    # # print(matrix)
    # #
    # # print(pd.DataFrame(matrix))
    # # matrix.columns =['stress_levels', 'old stressScores','hf_lf']
    # # # Change the row indexes
    # # df.index = ['Row_1', 'Row_2', 'Row_3', 'Row_4']
    #
    # print("new stress score and winRR", V1,"\n",
    # "new stress score and hf_lf", V2, "\n",
    # # "new stress score and RMSSD",V3,"\n",
    # "new stress score and old stressScore df.stressScores, stress_levels", V4, '\n',
    # "new stress score and lf", V5, '\n')
    #


    def spcoef(stress, variable):
        a, b = spearmanr( stress_levels, df.nn_interval)
        return a

    sp = {'Variable':['nn interval', 'hfnu', 'hf', 'lfnu', 'lf', 'hf_lf', 'rmssd'],
        'new_stress_score':[spcoef(stress_levels, df.nn_interval), spcoef(stress_levels, df.hfnu), spcoef(stress_levels, df.hf), spcoef(stress_levels, df.lf),spcoef(stress_levels, df.lfnu),spcoef(stress_levels, df.hf_lf),spcoef(stress_levels, df.RMSSD)],
        'old_stress_score':[spcoef(df.stressScores, df.nn_interval), spcoef(df.stressScores, df.hfnu), spcoef(df.stressScores, df.hf), spcoef(df.stressScores, df.lfnu),spcoef(df.stressScores, df.lf),spcoef(df.stressScores, df.hf_lf),spcoef(df.stressScores, df.RMSSD)]}
    # Create DataFrame
    df_n = pd.DataFrame(sp)
    print(df_n)











    coef, p = spearmanr( stress_levels, df.nn_interval)
    # print(typeof(spearmanr( stress_levels, df.nn_interval))


    print('Spearmans correlation coefficient for new stress and nn interval: %.3f' % coef )
    coef, p = spearmanr(df.stressScores, df.nn_interval)
    print('Spearmans correlation coefficient for old stress and nn interval: %.3f' % coef )
    coef, p = spearmanr(stress_levels, df.hfnu)
    print('Spearmans correlation coefficient for hfnu: %.3f' % coef )
    coef, p = spearmanr(stress_levels, df.SDNN)
    print('Spearmans correlation coefficient for SDNN: %.3f' % coef)
    oef, p = spearmanr(stress_levels, df.RMSSD)
    print('Spearmans correlation coefficient for RMSSD: %.3f' % coef )
    coef, p = spearmanr(stress_levels, df.hf_lf)
    print('Spearmans correlation coefficient FOR hflf: %.3f' % coef )
    coef, p = spearmanr(stress_levels, df.lfnu)
    print('Spearmans correlation coefficient for lfnu: %.3f' % coef )
    coef, p = spearmanr(stress_levels, df.stressScores)
    print('Spearmans correlation coefficient for stressScores: %.3f' % coef )


    # interpret the significance
    alpha = 0.05
    if p > alpha:
        print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
        print('Samples are correlated (reject H0) p=%.3f' % p)
