from datetime import timedelta
import numpy as np
import json
import random

# Private libraries ###
from movesense_read import Movesense

file = open(r'C:\Users\madhu\Documents\NLSAP_Stress\BLEData_Kamini Balaji_2021_03_22.json')
input_data_frame = json.load(file)

def execute_algorithm(data, fs=512):

    assert len(data['captured_data']['hr']['HR in BPM']) != 0, "No HR recorded. A possible reason is poor belt contact."
    assert int(data['captured_data']['hr']['ticks'][-1]) // 512 > 60 * 10, "Recorded HR shows less than 10 minutes of data recording. This could either be due to incorrect data collection or poor electrode contact."


    is_data_dependent_dur = False
    is_sleep_cluster = True

    ticks = data['captured_data']['act']['ticks']
    step_count = data['captured_data']['act']['step count']
    act_level = data['captured_data']['act']['activity level']

    movesense_obj = Movesense(data, fs)
    starting_time = movesense_obj.time_in_dt

    starting_time = starting_time + timedelta(minutes=2)
    movesense_obj.obtain_hrv_features()
    hr = movesense_obj.windowed_hr

    assert sum(np.asarray(movesense_obj.hf_lf) == None) != len(movesense_obj.hf_lf),  "Erroneous HR recorded throughout. A possible reason is poor belt contact or hardware failure."
    if not(len(movesense_obj.hf_lf)) < 5:

        movesense_obj.tagging_motion_data()
        stress_score_threshold = movesense_obj.get_stress_region()
        step_count = movesense_obj.step_count
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
    output_json['stressScores'] =  [str(stress_region[i]) for i in range(len(list(stress_region)))]
    output_json['stepCount'] =  [str(step_count[i]) for i in range(len(list(step_count)))]
    output_json['recoveryLabel'] = [str(recovery_region[i]) for i in range(len(list(recovery_region)))]
    output_json['nn_interval'] = [str(movesense_obj.windowed_nn_intervals[i]) for i in range(len(movesense_obj.windowed_nn_intervals))]
    output_json['processedHr'] = [str(hr[i]) for i in range(len(list(hr)))]
    output_json['commonTime'] = [(starting_time + timedelta(minutes = i)).isoformat()+'Z' for i in range(len(stress_region))]
    output_json['sleepTime'] = [(starting_time + timedelta(seconds = i*30)).isoformat()+'Z' for i in range(len(sleep_scores))]
    output_json['plotTimeData'] = [output_json['commonTime'][0]]
    output_json['sleepQuality'] = [str(sleep_scores[i]) for i in range(len(sleep_scores))]
    output_json['activityLevel'] = [str(activity_region[i]) for i in range(len(list(activity_region)))]
    print('The stored json fields are:',output_json.keys())


    return output_json
output_json = execute_algorithm(input_data_frame)
with open('y.json', 'w') as outfile:
    json.dump(output_json, outfile)
