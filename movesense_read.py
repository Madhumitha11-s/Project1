import json
from datetime import datetime
import math
import statistics as st

import numpy as np
from scipy import interpolate
from tqdm import tqdm
import pandas as pd

from collections import Counter

from hrv_analysis_mod.extract_features import get_time_domain_features, get_frequency_domain_features, _create_interpolation_time, \
    _create_time_info
#  AuraHealthcare Toolkit
from hrv_analysis_mod.preprocessing import remove_outliers, remove_ectopic_beats, interpolate_nan_values
print("Reading movesense_read")
class Movesense:

    def __init__(self, data, list_of_name,date, fs, isjson=True):

        self.list_of_name = list_of_name

        self.date = date

        self.fs = fs
        self.data = data
        self.isjson = isjson

        self.t_sc = np.array([])
        self.sc = np.array([])
        self.act_category = np.array([])
        self.st_time = 0
        self.time_in_hrf = 0
        self.erroneous_count = 0
        if self.isjson:
            ### Returns RR interval duration in ms
            self.rr = self.json_reader()
        else:
            print('Please provide path to a json file')

        self.windowed_hr = np.array([])
        self.hf = np.array([])
        self.lf = np.array([])
        self.hf_lf = np.array([])
        self.hfnu = np.array([])
        self.lfnu =np.array([])
        self.psd = np.array([])


        self.hr = np.array([])
        self.interpolate_fs = 4
        self.process_window = 5  ###(5 minute window)
        self.window_size = self.interpolate_fs * 60 * self.process_window

        self.stress_regions = np.array([])
        self.sleep_hr = np.array([])
        self.windowed_nn_intervals = np.array([])

        self.window_slide = 1  ### Overlap in minutes
        self.stride_samples = self.window_slide * 60 * self.interpolate_fs  ### window_size - overlap_samples)

        self.win_RR = np.array([])
        self.RMSSD = np.array([])

    def recompute_ticks_rr(self, rr_in_ms):

        ### Will only factor for no HR if no HR is received for over 20 seconds ###
        no_hr_indices = np.where(rr_in_ms > 20000)
        indice_shift = 0
        for i in list(no_hr_indices[0]):
            self.tsc_rr = np.insert(self.tsc_rr, i + indice_shift , (self.tsc_rr[i-1 + indice_shift] + 2100))
            self.tsc_rr = np.insert(self.tsc_rr, i + 1 + indice_shift , (self.tsc_rr[i + indice_shift] + 2100))
            self.tsc_rr = np.insert(self.tsc_rr, i + 2 + indice_shift , (self.tsc_rr[i + 1 + indice_shift] + 2100))
            indice_shift += 3
        rr_in_ms = np.asarray([int( (self.tsc_rr[i] - self.tsc_rr[i-1]) / self.fs * 1000) for i in range(1, len(self.tsc_rr))])
        rr_in_ms =  list(np.concatenate((np.zeros(1) ,rr_in_ms)))
        return rr_in_ms

    def json_reader(self):


        #
        # with open(self.data, 'r') as infile:
        #    data = json.load(infile)

        data = self.data
        self.tsc_rr = data['captured_data']['hr']['ticks']
        rr_in_ms = data['captured_data']['hr']['RR in ms']
        self.hr1 = data['captured_data']['hr']['HR in BPM']
        self.tsc_rr = [int(tick) for tick in self.tsc_rr]
        rr_in_ms = np.asarray([int( (self.tsc_rr[i] - self.tsc_rr[i-1]) / self.fs * 1000) for i in range(1, len(self.tsc_rr))])
        rr_in_ms =  np.concatenate((np.zeros(1) ,rr_in_ms))
        rr_in_ms = self.recompute_ticks_rr(rr_in_ms)
        self.hr1 = 60 // (np.asarray(rr_in_ms) / 1000)
        self.hr1[0] = 0

        self.sc = data['captured_data']['act']['step count']
        self.t_sc = data['captured_data']['act']['ticks']
        self.activity_level = data['captured_data']['act']['activity level']

        self.t_slp = data['captured_data']['slp']['ticks']
        self.position = np.asarray(data['captured_data']['slp']['sleep pos'])
        self.intensity = np.asarray(data['captured_data']['slp']['sleep moment'])

        # rr_in_ms = [int(rr_in_ms[i]) for i in range(len(rr_in_ms))]
        # self.hr1 = np.asarray([int(self.hr1[i]) for i in range(len(self.hr1))])

        self.sc = np.asarray([int(self.sc[i]) for i in range(len(self.sc))])
        self.tsc_rr = np.asarray([float(self.tsc_rr[i])/self.fs for i in range(len(self.tsc_rr))])
        self.t_sc = np.asarray([float(self.t_sc[i])/self.fs for i in range(len(self.t_sc))])
        self.t_slp = np.asarray([float(self.t_slp[i])/self.fs for i in range(len(self.t_slp))])
        self.activity_level = np.asarray([int(self.activity_level[i]) for i in range(len(self.activity_level))])
        self.time_in_string = data['Start_date_time']
        self.time_in_dt = datetime.strptime(self.time_in_string[:-1],"%Y-%m-%dT%H:%M:%S")

        return np.asarray(rr_in_ms)

    def obtain_hrv_features(self):

        rr_int = self.rr
        rr_times = _create_time_info(list(rr_int))

        # This remove outliers from signal
        rr_intervals_without_outliers = remove_outliers(rr_intervals=list(rr_int), verbose=False,
                                                        low_rri=270, high_rri=2000)
        # This replace outliers nan values with linear interpolation
        interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers,
                                                           interpolation_method="linear")

        # This remove ectopic beats from signal
        nn_intervals_list = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method="malik")
        # This replace ectopic beats nan values with linear interpolation
        interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)
        interpolated_nn_intervals = np.asarray(interpolated_nn_intervals)
        interpolated_nn_intervals[np.where(np.isnan(np.array(interpolated_nn_intervals)))] = 1000

        # ---------- Interpolation of signal ---------- #
        funct = interpolate.interp1d(x=rr_times, y=list(interpolated_nn_intervals), kind='linear')
        timestamps_interpolation = _create_interpolation_time(rr_times, self.interpolate_fs)

        nni_interpolation = funct(timestamps_interpolation)
        nn_interpolation = nni_interpolation

        all_hf_lf = []
        all_hf = []
        all_lf = []
        all_hfnu = []
        all_lfnu = []
        all_psd = []


        RMSSD = np.array([])


        SDNN = np.array([])  # Initiating array for storing RMSSD values for each window

        total_windows = int(timestamps_interpolation[-1] * self.interpolate_fs - self.window_size) // int(
            self.stride_samples) + 1

        assert total_windows > 0, "A minimum of 5 minutes of data collection is required"

        ### Rejection windows which have a lot of motion ###
        motion_time = self.t_sc[self.sc > 10] - 10
        for step_time in motion_time:
            nn_interpolation[(timestamps_interpolation >= step_time) & (timestamps_interpolation < step_time +10 ) ] = np.nan

        print("The total no of windows in hrv_analysis is:", total_windows)
        for i in tqdm(range(total_windows)):

            windowed_nni_intervals = nni_interpolation[
                                    self.stride_samples * i: self.stride_samples * i + self.window_size]

            windowed_nn_intervals = nn_interpolation[self.stride_samples * i : self.stride_samples* i + self.window_size]
            windowed_nn_intervals = windowed_nn_intervals[~np.isnan(windowed_nn_intervals)]

            if not(sum(windowed_nn_intervals != 1000)): ### Accounting for erroneous RR-Intervals
                RMSSD = np.append(RMSSD, math.nan)
                SDNN = np.append(SDNN, math.nan)


                self.windowed_hr = np.append(self.windowed_hr, 0)
                all_hf_lf.append(None)
                all_hfnu.append(None)
                all_lfnu.append(None)
                all_hf.append(None)
                all_lf.append(None)
                all_psd.append(None)

                self.erroneous_count += 1
                self.windowed_nn_intervals = np.append(self.windowed_nn_intervals, 0)
                self.win_RR = np.append(self.win_RR, 0)

            else:

                freq_domain_features = get_frequency_domain_features(windowed_nn_intervals)
                time_domain_features = get_time_domain_features(windowed_nn_intervals)



                RMSSD = np.append(RMSSD, time_domain_features['rmssd'])
                SDNN = np.append(SDNN, time_domain_features['sdnn'])


                # sdnn = np.append(sdnn, time_domain_features['sdnn'])
                # sdsd = np.append(sdsd, time_domain_features['sdsd'])
                HF = freq_domain_features['hf']
                LF = freq_domain_features['lf']
                HF_LF = freq_domain_features['hf'] / freq_domain_features['lf']
                HFnu = freq_domain_features['hf'] / (freq_domain_features['lf'] + freq_domain_features['hf'])
                LFnu = freq_domain_features['lf'] / (freq_domain_features['lf'] + freq_domain_features['hf'])
                # PSD = freq_domain_features['psd']


                if np.isnan(HF_LF):
                    HF_LF = None
                    HFnu = None
                    LFnu = None
                    HF = None
                    LF = None
                    # PSD = None

                all_hf_lf.append(HF_LF)
                all_hf.append(HF)
                all_lf.append(LF)
                all_hfnu.append(HFnu)
                all_lfnu.append(LFnu)
                # all_psd.append(PSD)

                if HF_LF != None:
                    self.windowed_hr = np.append(self.windowed_hr, np.mean(60 * 1000 / windowed_nn_intervals))
                    self.windowed_nn_intervals = np.append(self.windowed_nn_intervals, np.mean(windowed_nn_intervals))
                    self.win_RR = np.append(self.win_RR, np.mean(windowed_nn_intervals))

                else:
                    self.windowed_hr = np.append(self.windowed_hr, 0)
                    self.windowed_nn_intervals = np.append(self.windowed_nn_intervals, 0)
                    self.win_RR = np.append(self.win_RR, 0)


        #
        # import csv
        # field_names = ['mean_nni', 'sdnn', 'sdsd', 'nni_50', 'pnni_50', 'nni_20', 'pnni_20', 'rmssd', 'median_nni', 'range_nni', 'cvsd', 'cvnni', 'mean_hr', 'max_hr', 'min_hr', "std_hr"]
        #
        # with open('time_domain_features.csv', 'a') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames = field_names)
        #     writer.writerows(time_domain_features)

        self.hf_lf = all_hf_lf
        self.hf = all_hf
        self.lf = all_lf
        self.hfnu = all_hfnu
        self.lfnu = all_lfnu
        # self.psd = all_psd


        self.RMSSD = RMSSD
        self.SDNN = SDNN

        # self.sdsd = sdsd

        # hf_lf_data = {'hf_lf': self.hf_lf,'hf': self.hf, 'lf': self.lf}
        #
        #
        #
        # hf_lf_data = pd.DataFrame(hf_lf_data, columns=['hf_lf','hf','lf'])
        # hf_lf_data.to_csv (r'C:\Users\madhu\Documents\NLSAP_Stress\data\hf_lf_data.csv', index = False, header=True)

        # pd.DataFrame(self.hf_lf).T.to_csv('hf_lf.csv')
        # pd.DataFrame(self.hf).T.to_csv('hf.csv')
        # pd.DataFrame(self.lf).T.to_csv('lf.csv')
        # pd.DataFrame(self.psd).T.to_csv('psd.csv')
        #
        # pd.DataFrame(self.RMSSD).T.to_csv('RMSSD.csv')
        # pd.DataFrame(self.SDNN).T.to_csv('SDNN.csv')





    def get_stress_region(self):



        """ This function returns the threshold ranges for no, low, medium and high stress derived from HF/LF ratio
            The logic implementation is drived from percentile idea
        Arguments:
            plot {int} -- takes either value 0 or 1
                if plot equal to 1 -- plot the values
                otherwise -- plotting section is ignored
        """
        threshold_bounds = np.array([])

        # For HF/LF Normalization

        stress_score1 = np.asarray(self.hf_lf)


        stress_score1[(self.step_count>10) * (self.activity_intensity != 0) * (stress_score1 != None)] = -10
        stress_score1[stress_score1 == None] = -100
        stress_score1.tofile('stress_score1.csv', sep = ',')


        norm_stress_score1 = stress_score1/np.max(stress_score1)
        no_rr_val = 1 - (-100 / np.max(stress_score1))
        physical_act_val = 1 - (-10 / np.max(stress_score1))
        norm_stress_score1 = np.asarray([1-norm_stress_score1[i] for i in range(len(norm_stress_score1))])
        norm_stress_score1[norm_stress_score1 == physical_act_val] = 1.25
        norm_stress_score1[norm_stress_score1 == no_rr_val] = 1.50

        # for RR normalization
        stress_score2 = np.asarray(self.win_RR)
        stress_score2[(self.step_count>10) * (self.activity_intensity != 0) * (stress_score2 != 0)] = -10
        stress_score2[stress_score2 == None] = -100
        norm_stress_score2 = stress_score2/np.max(stress_score2)
        no_rr_val = 1 - (-100 / np.max(stress_score2))
        physical_act_val = 1 - (-10 / np.max(stress_score2))
        norm_stress_score2 = np.asarray([1-norm_stress_score2[i] for i in range(len(norm_stress_score2))])
        norm_stress_score2[norm_stress_score2 == physical_act_val] = 1.25
        norm_stress_score2[norm_stress_score2 == no_rr_val] = 1.50
        norm_stress_score = (norm_stress_score1 + norm_stress_score2)/2

        ### Logic to compute threshold for stress ranges ###

        physical_nohr_window_indices = np.where( (stress_score1 == -10) + (stress_score1 == -100) )


        mental_stress_score1 = np.delete(stress_score1, np.asarray(physical_nohr_window_indices))
        mental_stress_score2 = np.delete(stress_score2, np.asarray(physical_nohr_window_indices))

        norm_mental_stress_score1 = mental_stress_score1/np.max(mental_stress_score1)
        norm_mental_stress_score2 = mental_stress_score2/np.max(mental_stress_score2)

        norm_mental_stress_score = (norm_mental_stress_score1 + norm_mental_stress_score2)/2

        norm_mental_stress_score_sorted2 = np.sort(norm_mental_stress_score)


        mental_stress_bound2 = np.mean(norm_mental_stress_score_sorted2)

        ### Defining thresholds for stress ranges

        norm_mental_stress_score_sorted1 = norm_mental_stress_score_sorted2[norm_mental_stress_score_sorted2 < mental_stress_bound2]
        norm_mental_stress_score_sorted3 = norm_mental_stress_score_sorted2[norm_mental_stress_score_sorted2 > mental_stress_bound2]
        mental_stress_bound1 = np.mean(norm_mental_stress_score_sorted1)
        mental_stress_bound3 = np.mean(norm_mental_stress_score_sorted3)

        medium_stress_high_bound = 1- mental_stress_bound1
        low_stress_high_bound = 1- mental_stress_bound2
        no_stress_high_bound = 1- mental_stress_bound3

        for element in norm_stress_score:

            if element <= no_stress_high_bound:
                self.stress_regions = np.append(self.stress_regions, 0)
            elif element > no_stress_high_bound and element <= low_stress_high_bound:
                self.stress_regions = np.append(self.stress_regions, 1)
            elif element > low_stress_high_bound and element <= medium_stress_high_bound:
                self.stress_regions = np.append(self.stress_regions, 2)
            elif element > medium_stress_high_bound and element <= 1:
                self.stress_regions = np.append(self.stress_regions, 3)
            elif element == 1.25:
                self.stress_regions = np.append(self.stress_regions, 4)
            elif element == 1.50:
                self.stress_regions = np.append(self.stress_regions, 5)
        self.stress_levels = self.stress_regions

        threshold_bounds = np.array([0, no_stress_high_bound, low_stress_high_bound, medium_stress_high_bound, 1])
        as_list = threshold_bounds.tolist()









        # name_threshold_bounds = np.array([])
        # name_threshold_bounds = np.array([self.list_of_name[1], self.date, 0, no_stress_high_bound, low_stress_high_bound, medium_stress_high_bound, 1])
        # #pd.DataFrame(threshold_bounds).T.to_csv('threshold_bounds.csv')
        # import csv
        #
        # with open(r'C:\Users\madhu\Documents\NLSAP_Stress\data\threshold_bounds', 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(name_threshold_bounds)



        return threshold_bounds

    def duration_dependent_stress_bins(self):

        """ This function returns the threshold ranges for no, low, medium and high stress derived from HF/LF ratio
            The logic implementation is drived from percentile idea
        Arguments:
            plot {int} -- takes either value 0 or 1
                if plot equal to 1 -- plot the values
                otherwise -- plotting section is ignored
        """
        ## Duration dependent allotment of stress bins --  Estimated -- 5 bins per hour
        total_duration = math.floor(len(self.stress_levels)/56)
        defined_stress_levels = []
        # defined_stress_regions = []
        motion_indices = np.where(self.stress_levels == 4)
        no_HR_indices = np.where(self.stress_levels == 5)

        if total_duration > 0 and total_duration <= 8:
            ## CASE 1 : where the total duartion is between 1 - 8 hrs
            ## Here 4 indicates the per hour stress values will be averaged for every 4 mins --> 14 bins
            slide_by_samples = 4


        elif total_duration > 8 and total_duration <= 16:
            ## CASE 2 : where the total duartion is between 8 - 16 hrs
            ## Here 7  indicates the per hour stress values will be averaged for every 7 mins --> 8 bins
            slide_by_samples = 7


        elif total_duration > 16 and total_duration <= 30:
            ## CASE 3 : where the total duartion is between 16 - 24 hrs
            ## Here 14 indicates the per hour stress values will be averaged for every 14 mins --> 4 bins
            slide_by_samples = 14

        for i in range(0, len(self.stress_levels), slide_by_samples):
            windowed_stress_levels = self.stress_levels[i:i+slide_by_samples]
            print(window_stress_level,"imhere")
            windowed_stress_levels[np.where(windowed_stress_levels == 4)] = 0
            windowed_stress_levels[np.where(windowed_stress_levels == 5)] = 4

            # window_stress_level = math.floor(np.mean(windowed_stress_levels))
            try:
                window_stress_level = st.mode(windowed_stress_levels) ###This gives out an error
            except:

                ### If there are multiple modes, the min value is considered as stress score
                counts = st._counts(windowed_stress_levels)
                window_stress_level = min([values for values,count in counts])

            mode_stress_levels = np.full(shape = len(windowed_stress_levels), fill_value = window_stress_level, dtype = np.int)
            defined_stress_levels = np.append(defined_stress_levels, mode_stress_levels)



        self.stress_regions = defined_stress_levels
        self.stress_regions[np.array(motion_indices)] = 5
        self.stress_regions[np.array(no_HR_indices)] = 4

    def get_recovery_region(self):

        '''
        Function to classify data based on stress-recovery levels into :
            1> Low Recovery
            2> Moderate Recovery
            3> High Recovery
        based on the RMSSD HRV parameter.

        Input:
            1-d array of RMSSD values.

        Output:
            1-d array of stress-recovery values(Low,Moderate,High) corresponding to input values.

        '''
        nans, x = self.nan_helper(self.RMSSD)  # Using defined function for Removing nan values
        self.RMSSD[nans] = np.interp(x(nans), x(~nans), self.RMSSD[~nans])  # Interpolating nan values

        # This removes DC component
        RMSSD_0avg = self.RMSSD - np.mean(self.RMSSD)

        # Cubing of data values for better classification
        RMSSD_cube = (RMSSD_0avg) ** 3
        RMSSD_cube = RMSSD_0avg

        ##Running mean Filter for smoothening

        n = len(self.RMSSD)
        RMSSD_filt = np.zeros(n)

        # implement the running mean filter
        k = int(0.01 * n)  ## 1% of the total length # filter window is actually k*2+1
        if k < 1:
            k = 1
        for i in range(k, n - k):
            # each point is the average of k surrounding points
            RMSSD_filt[i] = np.mean(RMSSD_cube[i - k:i + k])

        ## Thresholds for classification
        t1 = np.mean(RMSSD_filt)
        # threshold2
        t2 = np.mean(RMSSD_filt) + np.std(RMSSD_filt)

        stress_recovery = np.array([])

        # This applies classification logic to the data
        for i in range(len(RMSSD_filt)):
            if self.stress_regions[i] == 5:
                stress_recovery = np.append(stress_recovery,3)
            elif RMSSD_filt[i] < t1:
                stress_recovery = np.append(stress_recovery,0)
            elif ((RMSSD_filt[i] >= t1) and (RMSSD_filt[i] < t2)) and ((self.stress_regions[i]==0) or (self.stress_regions[i]==1)):
                stress_recovery = np.append(stress_recovery,1)
            elif RMSSD_filt[i] >= t2 and ((self.stress_regions[i]==0) or (self.stress_regions[i]==1)):
                stress_recovery = np.append(stress_recovery,2)
            else:
                stress_recovery = np.append(stress_recovery,0)

        stress_recovery[self.hf_lf == None] = 3
        return stress_recovery

    def nan_helper(self, y):

        """Helper to handle indices and logical indices of NaNs.
        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
            to convert logical indices of NaNs to 'equivalent' indices
        """

        return np.isnan(y), lambda z: z.nonzero()[0]

    def get_sleep_stage(self):

        # #Sleep staging using position, intensity and HR
        # if not(self.islatest_data_collection):
        #     total_seconds = self.peak_locs[-1] // self.fs
        # else:
        #     total_seconds = np.cumsum(self.peak_locs)[-1] // 1000

        total_seconds = np.cumsum(self.rr)[-1] // 1000
        self.hr1 = np.array(self.hr1)
        self.tsc_rr = np.array(self.tsc_rr)

        #Window of 1 minute
        seconds_per_window = 60
        total_windows = int((total_seconds - seconds_per_window) // seconds_per_window + 1)
        print("The total no of windows in sleep data is {}, total seconds is {} and seconds per window is {}:".format(total_windows, total_seconds, seconds_per_window))
        self.slp_category = np.array([]) ### Stores either 'Activity' or 'Sleep'
        stage = np.zeros(total_windows)

        req_int_all = []
        req_pos_all= []
        req_hr_all =[]

        for i in tqdm(range(int(total_windows))):

            req_intensity = self.intensity[(self.t_slp > i * seconds_per_window ) & (self.t_slp < i * seconds_per_window + seconds_per_window) == True]
            req_pos_level = self.position[(self.t_slp > i * seconds_per_window ) & (self.t_slp < i * seconds_per_window + seconds_per_window) == True]
            req_hr = self.hr1[(self.tsc_rr > i * seconds_per_window) & (self.tsc_rr < i * seconds_per_window + seconds_per_window) == True ]

            req_int_all.append(req_intensity)
            req_pos_all.append(req_pos_level)
            req_hr_all.append(req_hr)

            if (req_intensity.size == 0) or (req_hr.size == 0) or (req_pos_level.size == 0):
                self.slp_category = list(np.append(self.slp_category,'Activity'))

            else:
                self.slp_category = list(np.append(self.slp_category, 'Sleep'))

        #### Get sleep indices
        index_pos_list = []
        index_pos = 0

        while True:
            try:
                # Search for 'Sleep' in list from indexPos to the end of list
                index_pos = self.slp_category.index('Sleep', index_pos)
                # Add the index position in list
                index_pos_list.append(index_pos)
                index_pos += 1
            except ValueError as e:
                break

        ##### Store only those indices where there are greater than 30 consecutive windows of sleep
        self.sleep_hr = np.empty(total_windows, dtype= object)
        self.sleep_int = np.empty(total_windows, dtype = object)
        self.sleep_pos = np.empty(total_windows, dtype = object)

        if len(index_pos_list) != 0: ### Checking if sleep exists

            res = [[index_pos_list[0]]]
            for i in range(1, len(index_pos_list)):
                if index_pos_list[i-1]+1 == index_pos_list[i]:
                    res[-1].append(index_pos_list[i])

                else:
                    res.append([index_pos_list[i]])
            new_list =[]
            #Remove sleep indices with less than 30 minutes of sleep data, classify them as wake
            for list1 in res:
                if len(list1) >= 30:
                    new_list.append(list1)
            #### Sleep staging is performed only on the indices in new_list
            for i in range(0, len(new_list)):
                x = new_list[i]
                #print(x)
                for j in range(0, len(x)):
                    self.sleep_hr[x[j]] = (req_hr_all[x[j]])
                    self.sleep_pos[x[j]] = (req_pos_all[x[j]])
                    self.sleep_int[x[j]]= (req_int_all[x[j]])
            i = 0
            while i <(len(self.sleep_hr)):
                if self.sleep_hr[i] is not None:
                    hr_prev = np.mean(self.sleep_hr[i])
            #print(hr_prev)
                    pos_prev = len(list(Counter(self.sleep_pos[i]).values()))
                    int_prev = len(list(Counter(self.sleep_int[i]).values()))
                    break
                else:
                    i+=1
                    continue
            ### Sleep classification
            if len(new_list) != 0: ### If length of new list is zero, all of sleep is wake stage.
                if hr_prev > 60: #Classifying the first one min as wake stage since more likely for the person to be awake at the start
                    stage[0] = 0
                else:
                    stage[0] = 1
        cnt = 0
        lim = int(0.22 * len(self.sleep_pos))
        for i in tqdm(range(1, len(self.slp_category))):
            if self.slp_category[i] == 'Activity':
                stage[i] = 4
            elif self.slp_category[i] == 'Sleep':
                if self.sleep_hr[i] is not None:
                    hr_curr = sum(self.sleep_hr[i])/ len(self.sleep_hr[i])

                    pos_cur = len(list(Counter(self.sleep_pos[i]).values()))
                    int_cur = len(list(Counter(self.sleep_int[i]).values()))

                    ### If activity occured within 20 minutes prior to lying down, it will automatically be labelled as Wake
                    if i > 20:
                        if sum(np.asarray(self.slp_category[i-20:i]) == 'Activity'):
                            stage[i] == 0
                            continue
                    ### Actual algo starts here

                    if stage[i - 1] == 0:

                        if hr_curr > 70:
                            stage[i] = 0
                        else:
                            if int_cur <= int_prev:
                                stage[i] =1
                    elif stage[i - 1] == 1:
                        if cnt <= lim and hr_curr < 55:
                            cnt += 1
                            stage[i] = 2
                        elif hr_curr - hr_prev < 0:
                            if pos_cur >= 2 or hr_prev - hr_curr <= 10 or int_cur > 3:
                                stage[i] =1
                            else:
                                if cnt <= lim:
                                    cnt += 1
                                    stage[i] = 2
                                else:
                                    stage[i]=1

                        elif hr_curr - hr_prev > 0:
                            if pos_cur > pos_prev or hr_curr > 70:
                                stage[i] =0
                            elif hr_curr - hr_prev >= 8 or pos_cur < pos_prev or int_cur < int_prev:
                                stage[i] = 3
                            else:
                                stage[i] = 1
                        else:
                            stage[i]=3
                    elif stage[i - 1] == 2:
                        if cnt <= lim and hr_curr < 62:
                            cnt += 1
                            stage[i]=2
                        elif pos_cur > 1 and int_cur > 3:
                            stage[i]=0
                        else:
                            stage[i]=1
                    elif stage[i - 1] == 3:
                        if hr_curr < hr_prev or pos_cur > pos_prev or int_cur > int_prev:
                            stage[i] =1
                        else:
                            stage[i] = 3

                    hr_curr = hr_prev
                    pos_cur = pos_prev
                    int_cur = int_prev
                else:
                    stage[i] = 0

        stage = stage[2:-2]
        self.slp_category = self.slp_category[2:-2]
        self.sleep_hr = self.sleep_hr[2:-2]
        return stage

    def obtain_sleep_stages_cluster(self):

        ###Sleep staging using clustering of HR values
        seconds_per_window = 60

        ###Store mean of each window of sleep_hr, or nan if it is activity or less than 30 mins of sleep
        new_hr =[]
        for i in range(0, len(self.sleep_hr)):
            if self.sleep_hr[i] is not None:
                new_hr.append(np.mean(self.sleep_hr[i]))
            else:
                new_hr.append(np.nan)

        ####generate sleep stages using Heart Rate values and based on global statisctics of sleep architecture

        av_hr= np.nanmean(new_hr)
        n_hr=np.sort(new_hr)
        #max_hr=n_hr[-1]
        #min_hr=n_hr[0]
        max_hr =np.nanmax(n_hr)
        min_hr=np.nanmin(n_hr)

        cluster1=[]
        cluster2=[]
        cluster3=[]
        cluster4=[]

        wake=[]
        deep_sleep=[]
        light_sleep=[]
        rem=[]
        nrem=[]

        key1=[]
        key2=[]
        keyd=[]
        key3=[]
        keyw=[]
        keyr=[]
        keyn=[]
        keyl=[]


        scorew=[]
        scorer=[]
        scoren=[]
        scorel=[]
        scored=[]

        ####sort and segregate Heart rate values

        for i in range(0,len(new_hr)):

            if new_hr[i]>=min_hr and new_hr[i]<av_hr :
                cluster1=np.append(cluster1, new_hr[i])
                key1=np.append(key1,i)

        mean_c1=np.nanmean(cluster1)

        for j in range(0,len(new_hr)):
            if new_hr[j]>= min_hr and new_hr[j]<=mean_c1:
                cluster2=np.append(cluster2, new_hr[j])
                key2=np.append(key2,j)

        mean_c2=np.nanmean(cluster2)

        ####lower HR values --> annotate as deep sleep
        for k in range(0,len(new_hr)):
            if new_hr[k]>= min_hr and new_hr[k]<= mean_c2:
                deep_sleep=np.append(deep_sleep, new_hr[k])
                keyd=np.append(keyd,k)
                scored=np.append(scored,2)

        for m in range(0, len(new_hr)):
            if new_hr[m]>=av_hr and new_hr[m]<=max_hr:
                cluster3=np.append(cluster3, new_hr[m])
                key3=np.append(key3,m)

        mean_c3=np.nanmean(cluster3)

        for n in range(0, len(new_hr)):

            if new_hr[n]>av_hr and new_hr[n]<=mean_c3:
                cluster4=np.append(cluster4,new_hr[n])

        mean_c4=np.nanmean(cluster4)

        ####higher HR values --> annotate as Wake
        for u in range(0, len(new_hr)):

            if new_hr[u]>=mean_c4 and new_hr[u]<=max_hr:
                wake=np.append(wake, new_hr[u])
                keyw=np.append(keyw,u)
                scorew=np.append(scorew,0)
        #### REM sleep --> 3
            elif new_hr[u]>av_hr and new_hr[u]<mean_c4:
                rem=np.append(rem,new_hr[u])
                keyr=np.append(keyr,u)
                scorer=np.append(scorer,3)

        #### NREM --> 1

            elif new_hr[u]>mean_c2 and new_hr[u]<=av_hr:
                nrem=np.append(nrem,new_hr[u])
                keyn=np.append(keyn,u)
                scoren=np.append(scoren,1)

        ### Light sleep --> 1 (NREM1, NREM2, REM)

            elif new_hr[u]>mean_c2 and new_hr[u]<mean_c4:
                light_sleep=np.append(light_sleep,new_hr[u])
                keyl=np.append(keyl,u)
                scorel=np.append(scorel,1)

        d_sleep=np.sort(deep_sleep)
        nrem_sleep=nrem

        #### screened based on global statistics of Deep sleep i.e. cannot exceed 20% of total duration
        for t in range(0,len(deep_sleep)):
            ratio_deep= len(d_sleep)/len(new_hr)

            if  ratio_deep>0.20:
                d_sleep=d_sleep[:-1]
                nrem_sleep=np.append(nrem_sleep,d_sleep[-1])

            else:
                deep_sleep = d_sleep
                nrem = nrem_sleep

        key=[]
        value=[]
        score=[]

        key=np.append(key,keyd)
        #key=np.append(key,keyl)
        key=np.append(key,keyn)
        key=np.append(key,keyr)
        key=np.append(key,keyw)

        value=np.append(value,deep_sleep)
        #value=np.append(value,light_sleep)
        value=np.append(value,nrem)
        value=np.append(value,rem)
        value=np.append(value,wake)

        score=np.append(score,scored)
        #score=np.append(score,scorel)
        score=np.append(score,scoren)
        score=np.append(score,scorer)
        score=np.append(score,scorew)

        key=list(key)
        value=list(value)
        score=list(score)

        gen_sleep_stage = {
            'KEY': key,
            'HR_VALUE': value,
            'SLEEP_STAGE':score}

        percentage_wake=len(wake)*100/len(new_hr)
        percentage_nrem=len(nrem)*100/len(new_hr)
        percentage_rem=len(rem)*100/len(new_hr)
        percentage_deep_sleep=len(deep_sleep)*100/len(new_hr)
        percentage_light_sleep=len(light_sleep)*100/len(new_hr)


        gen_sleep_percentage= {
            'PERCENTAGE_WAKE': percentage_wake,
            'PERCENTAGE_NREM':percentage_nrem,
            'PERCENTAGE_REM':percentage_rem,
            'PERCENTAGE_LIGHT_SLEEP': percentage_light_sleep,
            'PERCENTAGE_DEEP_SLEEP': percentage_deep_sleep}


        # Sort stages_cluster according to the keys and store in stages_new
        stages_cluster = gen_sleep_stage['SLEEP_STAGE']
        key = [int(s) for s in gen_sleep_stage['KEY']]
        stages_new = np.empty(len(new_hr))

        for i in range(len(key)):
            j = key[i]
            stages_new[j]= stages_cluster[i]


        ## Assign -1 to activity windows
        for i in range(0, len(self.slp_category)):
            if self.slp_category[i] == 'Activity':
                stages_new[i] = 4
            elif self.slp_category[i] == 'Sleep':
                if self.sleep_hr[i] is None:
                    stages_new[i] = 0
                else:
                    continue
            else:
                continue

        # Remove the following transitions: wake-REM, REM-wake, wake-deep, small activity windows during sleep
        for i in range(1, len(stages_new)):
                if (stages_new[i-1] ==0 and stages_new[i] ==3) or (stages_new[i-1] ==3 and stages_new[i] ==0) or (stages_new[i-1] ==0 and stages_new[i] ==2):
                    if self.sleep_hr[i] is not None and self.sleep_hr[i-1] is not None:
                        hr_curr = np.mean(self.sleep_hr[i])
                        pos_cur = len(list(Counter(self.sleep_pos[i]).values()))
                        int_cur = len(list(Counter(self.sleep_int[i]).values()))
                        hr_prev = np.mean(self.sleep_hr[i-1])
                        pos_prev = len(list(Counter(self.sleep_pos[i-1]).values()))
                        int_prev =  len(list(Counter(self.sleep_int[i-1]).values()))
                        if (stages_new[i-1] ==0 and stages_new[i] ==3):
                            if hr_curr >70:
                                stages_new[i] =0

                            else:
                                if int_cur <= int_prev:
                                    stages_new[i] =1


                        elif (stages_new[i-1] ==3 and stages_new[i] ==0):

                            if hr_curr < hr_prev or pos_cur > pos_prev or int_cur > int_prev:
                                stages_new[i] = 1
                            else:
                                stages_new[i] =3

                        elif (stages_new[i-1] ==0 and stages_new[i] ==2):

                            if hr_curr>70:
                                stages_new[i] =0
                            else:
                                if int_cur <=int_prev:
                                    stages_new[i] =1

                    else:
                        stages_new[i] = stages_new[i-1]

                elif (stages_new[i-1] ==1 and stages_new[i] ==-1 and -1 not in stages_new[i+10:i+20]) or (stages_new[i-1] ==2 and stages_new[i] ==-1) or (stages_new[i-1] ==3 and stages_new[i] == -1 and -1 not in stages_new[i+10:i+20]):
                    stages_new[i] = stages_new[i-1]


                else:
                    continue

        # List to store start values of each sleep cycle
        list_a=[]
        list_b=[]
        for i in range(0, len(stages_new)-1):
            if stages_new[i] ==-1 and stages_new[i+1]== 0:
                list_a.append(i+1)
        for i in range(0, len(list_a)):
            x = list_a[i]
            for j in range(x, len(stages_new)):
                if stages_new[j] == -1:
                    list_b.append(j)
                    break
        list_c=[]

        if len(list_a) >len(list_b):
            list_b.append(len(stages_new))

        for i in range(0, len(list_a)):

            x = list_a[i]
            y = list_b[i]

            if 1 in stages_new[x:y] or 2 in stages_new[x:y] or 3 in stages_new[x:y]:
                list_c.append([x,y])

        # REM in the first 90 minutes is changed to wake, light and deep sleep in a 20:48:32 ratio
        total_rem_windows = int(90* (60/seconds_per_window))
        rem_to_wake = int(total_rem_windows*0.20)
        rem_to_light = int(total_rem_windows*0.48)

        if len(list_c) > 0:
            for i in range(0, len(list_c)):
                p= list_c[i][0]
                for q in range(p, p+rem_to_wake):
                    if stages_new[q]==3:
                        stages_new[q] =0
                for r in range(p+rem_to_wake, p+rem_to_light):
                    if stages_new[r] ==3 or stages_new[r]==0:
                        stages_new[r] = 1
                for s in range(p+rem_to_light, p+total_rem_windows):
                    if stages_new[s] ==3:
                        stages_new[s] =2


        # Change wake stages-->REM if ratio of wake >0.10
        if len(list_c) > 0:
            for i in range(0, len(list_c)):
                p, q= list_c[i]
                for t in range(0, len(stages_new[p:q])):
                    stages_new=list(stages_new)
                    wake_ratio = stages_new[p:q].count(0)/len(stages_new[p:q])
                    if wake_ratio>0.10:
                        for r in range(p+150, q):
                            if stages_new[r] ==0:
                                stages_new[r] =3
                                break
                    else:
                        continue

        return stages_new

    def tagging_motion_data(self, resolution=1):

        """ Method used to tag motion affected windows
             Any 5 window holding step count value greater than 10 will be
             neglected and tagged as motion affected
        Arguments:
            Input
            resolution{float} -- step count values generated for every n seconds
            Output

            step_count{int} -- Integrated step count values values computed
            for every 5 mins slided by 1 minute
        """

        ### Step count values computed for every minutes slided by 1 minute
        step_count = []
        activity_intensity = []

        ### Window considering only till the last window of heart rate
        total_seconds = np.cumsum(self.rr)[-1] // 1000
        ### No of samples considered for every window
        seconds_per_window = resolution * 60
        ### No of samples slided for every window
        stride_in_sec = self.window_slide * 60

        total_windows = (total_seconds - seconds_per_window) // seconds_per_window + 1
        print("The total no of windows in motion data is {}, total seconds is {} and seconds per window is {}:".format(total_windows, total_seconds, seconds_per_window))
        for iter in tqdm(range(int(total_windows))):

            required_sc = self.sc[(self.t_sc > iter * seconds_per_window) & (self.t_sc < iter * seconds_per_window + seconds_per_window) == True]
            required_hr = self.hr1[(self.tsc_rr > iter * seconds_per_window) & (self.tsc_rr < iter * seconds_per_window + seconds_per_window) == True]
            required_actlev = self.activity_level[(self.t_sc > iter * seconds_per_window ) & (self.t_sc < iter * seconds_per_window + seconds_per_window) == True]

            if required_actlev.size == 0:
                self.act_category = np.append(self.act_category,3)
                activity_intensity = np.append(activity_intensity, 0)
            else:
                mean_act =  sum(required_actlev)/len(required_actlev)
                # try:
                    # mean_hr = sum(required_hr)/len(required_hr)
                # except:
                    # mean_hr = 0

                if mean_act < 10:
                    self.act_category = np.append(self.act_category,0)
                elif mean_act < 25:
                    self.act_category = np.append(self.act_category,1)
                else:
                    self.act_category = np.append(self.act_category,2)
                activity_intensity = np.append(activity_intensity, mean_act)

            if required_sc.size == 0:
                step_count = np.append(step_count, 0)
            else:
                step_count = np.append(step_count, sum(required_sc))


        ### Removing the first and the last parts of the code.
        self.step_count = step_count[2:-2]
        self.act_category = self.act_category[2:-2]
        self.activity_intensity = activity_intensity[2:-2]
