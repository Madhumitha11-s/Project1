from datetime import timedelta
import numpy as np
import json
import pandas as pd
import random
import csv
import matplotlib.pyplot as plt
# import seaborn as sns

# Private libraries ###
from movesense_read import Movesense

import os


name_file = date = list_of_name = list_n = []
data_value = np.empty(shape=[0, 5])
min_mean_max = np.array([])
datum = {'night_deepandlightsleep_data': {"min_win_RR":([]) ,"mean_win_RR":([]),"max_win_RR": ([]),
"min_hf_lf":([]) ,"mean_hf_lf":([]),"max_hf_lf": ([])}}
stress_thresholds_data = {"subject": ([]),
"win_RR_1":([]) ,"win_RR_2": ([]),
"hf_lf_1":([]) ,"hf_lf_2": ([])}

NLSAP_path = r'C:\Users\madhu\Documents\NLSAP_Stress'


directory = os.path.join(NLSAP_path, "dumps", "") #navigate to ble dump
for filename in os.listdir(directory):
    path = os.path.join(directory, filename)
    list_of_name = filename.split("_")
    list_n = list_of_name[4].split(".")
    subject_name = list_of_name[1]
    name_file.append(list_of_name[1])
    date = list_of_name[2]+"_"+list_of_name[3]+"_"+list_n[0]
    file = open(path)
    input_data_frame = json.load(file)

    or_path = os.path.join(NLSAP_path, "data", "")
    path = or_path + list_of_name[1] + '\\'



    def execute_algorithm(data, list_of_name, date, fs=512,):
        assert len(data['captured_data']['hr']['HR in BPM']) != 0, "No HR recorded. A possible reason is poor belt contact."
        assert int(data['captured_data']['hr']['ticks'][-1]) // 512 > 60 * 10, "Recorded HR shows less than 10 minutes of data recording. This could either be due to incorrect data collection or poor electrode contact."
        is_data_dependent_dur = False
        is_sleep_cluster = True

        ticks = data['captured_data']['act']['ticks']
        step_count = data['captured_data']['act']['step count']
        act_level = data['captured_data']['act']['activity level']

        movesense_obj = Movesense(data, list_of_name, date, fs)
        starting_time = movesense_obj.time_in_dt
        starting_time = starting_time + timedelta(minutes=2)
        movesense_obj.obtain_hrv_features()

        hr = movesense_obj.windowed_hr

        assert sum(np.asarray(movesense_obj.hf_lf) == None) != len(movesense_obj.hf_lf),  "Erroneous HR recorded throughout. A possible reason is poor belt contact or hardware failure."
        if not(len(movesense_obj.hf_lf)) < 5:

            movesense_obj.tagging_motion_data()
            stress_score_threshold = movesense_obj.get_stress_region()

            step_count = movesense_obj.step_count
            hf_lf_freq = movesense_obj.hf_lf
            hf_freq = movesense_obj.hf
            lf_freq = movesense_obj.lf
            RMSSD_freq = movesense_obj.RMSSD
            SDNN = movesense_obj.SDNN
            stress_region = movesense_obj.stress_regions


            if is_data_dependent_dur:
                movesense_obj.duration_dependent_stress_bins()


            recovery_region = movesense_obj.get_recovery_region()
            activity_region = movesense_obj.act_category
            sleep_scores =  movesense_obj.get_sleep_stage()


            sleep_len = len(sleep_scores[(sleep_scores != 4)])
            if sleep_len > 5 * 60 and is_sleep_cluster:
                sleep_scores = movesense_obj.obtain_sleep_stages_cluster()

            ### Removing No HR regions in all plots

            sleep_scores[stress_region == 5] = 5
            activity_region[stress_region == 5] = 4
            hr[stress_region == 5] = 0

        else:

            stress_region = []
            recovery_region = []
            activity_region = []
            sleep_scores = []

            movesense_obj.tagging_motion_data()
            step_count = movesense_obj.step_count
            for i in range(len(hr)):

                ### All no HR region ###
                stress_region.append(5)
                recovery_region.append(3)
                activity_region.append(4)
                sleep_scores.append(5)

        output_json = {}

        output_json['stressScores'] =  [(stress_region[i]) for i in range(len(list(stress_region)))]
        output_json['stepCount'] =  [(step_count[i]) for i in range(len(list(step_count)))]
        output_json['recoveryLabel'] = [(recovery_region[i]) for i in range(len(list(recovery_region)))]
        output_json['nn_interval'] = [(movesense_obj.windowed_nn_intervals[i]) for i in range(len(movesense_obj.windowed_nn_intervals))]
        output_json['hf_lf'] = [(movesense_obj.hf_lf[i]) for i in range(len(movesense_obj.hf_lf))]
        output_json['hfnu'] = [(movesense_obj.hfnu[i]) for i in range(len(movesense_obj.hfnu))]
        output_json['lfnu'] = [(movesense_obj.lfnu[i]) for i in range(len(movesense_obj.lfnu))]
        output_json['hf'] = [(movesense_obj.hf[i]) for i in range(len(movesense_obj.hf))]
        output_json['lf'] = [(movesense_obj.lf[i]) for i in range(len(movesense_obj.lf))]
        output_json["processedHr"] = [str(hr[i]) for i in range(len(list(hr)))]
        output_json['RMSSD'] = [(movesense_obj.RMSSD[i]) for i in range(len(movesense_obj.RMSSD))]
        output_json['SDNN'] = [(movesense_obj.SDNN[i]) for i in range(len(movesense_obj.SDNN))]
        output_json['win_RR'] = [(movesense_obj.win_RR[i]) for i in range(len(movesense_obj.win_RR))]
        output_json['commonTime'] = [(starting_time + timedelta(minutes = i)).isoformat()+'Z' for i in range(len(stress_region))]
        output_json['sleepTime'] = [(starting_time + timedelta(seconds = i*30)).isoformat()+'Z' for i in range(len(sleep_scores))]
        output_json['sleepQuality'] = [(sleep_scores[i]) for i in range(len(sleep_scores))]
        output_json['activityLevel'] = [(activity_region[i]) for i in range(len(list(activity_region)))]



        print('The stored json fields are:',output_json.keys())


        return output_json

    output_json = execute_algorithm(input_data_frame, list_of_name, date, 512)
    df = pd.DataFrame(output_json)
    clean_df = df[df.stressScores != 5.0]
    clean_df = clean_df[clean_df.stressScores != 4.0]


    def compute_parameters( df , type):

        df_NAN = df.replace(0, np.nan)
        df_NAN = df_NAN.replace(-10, np.nan)

        min_win_RR = df_NAN["win_RR"].min()
        mean_win_RR = df_NAN["win_RR"].mean()
        max_win_RR = df_NAN['win_RR'].max()

        min_hf_lf = df_NAN["hf_lf"].min()
        mean_hf_lf = df_NAN["hf_lf"].mean()
        max_hf_lf = df_NAN['hf_lf'].max()

        min_mean_max = np.array([
        list_of_name[1], date, type,
        min_win_RR,mean_win_RR,max_win_RR,
        min_hf_lf,mean_hf_lf,max_hf_lf])


        return min_mean_max

    def dict(mmm,datum_temp):
        mmm["min_win_RR"].append(str(datum_temp[3]))
        mmm["mean_win_RR"].append(str(datum_temp[4]))
        mmm["max_win_RR"].append(str(datum_temp[5]))
        mmm["min_hf_lf"].append(str(datum_temp[6]))
        mmm["mean_hf_lf"].append(str(datum_temp[7]))
        mmm["max_hf_lf"].append(str(datum_temp[8]))
        return mmm

    day_data = compute_parameters(clean_df, "24hours")


    #night_deep and light sleep data
    temp_data = clean_df[clean_df.sleepQuality != 0.0]
    night_deepandlightsleep_data = compute_parameters(temp_data[temp_data.sleepQuality != 3.0], "night_deepandlightsleep_data")
    dict(datum['night_deepandlightsleep_data'], night_deepandlightsleep_data)



    or_path = os.path.join(NLSAP_path, "movesense_report_json_output", "")
    temp_filename = str(subject_name)+"_" + str(date) + "_output_json.json"
    path = or_path + temp_filename
    with open( path, 'w') as outfile:
        json.dump(output_json, outfile)

    df = pd.read_json (path)
    or_path = os.path.join(NLSAP_path, "movesene_report_json_csv", "")
    temp_filename = str(subject_name) + "_"+ str(date)+"_output_json.csv"
    path = or_path + temp_filename
    df.to_csv (path , index=None)




df = pd.DataFrame(datum['night_deepandlightsleep_data'])
temp_filename = str(subject_name) + "_"+ str(date)+"night_deepandlightsleep_data.csv"
or_path = os.path.join(NLSAP_path, "temp_data/",temp_filename)
df.to_csv(or_path, index = False)
df = pd.read_csv(or_path)


def store_stress_thresholds_data(stress_thresholds_data,df,subject_name):
    stress_thresholds_data["subject"].append(subject_name)
    stress_thresholds_data["win_RR_2"].append(df["min_win_RR"].max())
    stress_thresholds_data["win_RR_1"].append(df["mean_win_RR"].max())
    stress_thresholds_data["hf_lf_1"].append(df["mean_hf_lf"].max())
    stress_thresholds_data["hf_lf_2"].append(df["min_hf_lf"].max())
    return stress_thresholds_data

stress_thresholds_data = store_stress_thresholds_data(stress_thresholds_data,df,subject_name)


or_path = os.path.join(NLSAP_path, "threshold_data", "")
temp_filename = str(subject_name) + "_stress_thresholds_data.json"
path = or_path + temp_filename
with open( path, 'w') as outfile:
    json.dump(stress_thresholds_data, outfile)

df = pd.read_json(path)
df.to_csv (or_path + "stress_thresholds_data.csv", index=None)
