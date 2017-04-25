# -*- coding: utf-8 -*-
import os
import sys
HOME = os.path.expanduser('~')
sys.path.append(os.path.join(HOME, "Dropbox", "dataset", "Scripts"))
from tianchi_api.features import *
from tianchi_api.system import getHome

HOME = getHome()
continuous_zero_filled_threshold = 1
consider_anomaly = True
lag = 10
mean_window_size = 3
std_window_size = 3
diff_gap_size = 3
output_length = 7
for shop in xrange(1, 2001):
    print shop
    save_folder = os.path.join(HOME, "Dropbox", "dataset", "Analysis", "PayTrend_Filled_threshold_%s_%s_anomaly" % (continuous_zero_filled_threshold, "consider" if consider_anomaly else "not_consider"), "training_allDefaultFeatures_lag%s_output%s"%(lag,output_length))
    try:
        os.makedirs(save_folder)
    except:
        pass
    save_file = os.path.join(save_folder, "feature_shop_%s.csv" %shop)
    get_shop_feature(shop,[get_lag_days(continuous_zero_filled_threshold=continuous_zero_filled_threshold,lag=lag),\
                           get_mean_lag_values(continuous_zero_filled_threshold=continuous_zero_filled_threshold,lag=lag,mean_window_size=mean_window_size), \
                           get_std_lag_values(continuous_zero_filled_threshold=continuous_zero_filled_threshold,lag=lag,std_window_size=std_window_size),\
                           get_mean_diff_lag_values(continuous_zero_filled_threshold=continuous_zero_filled_threshold,lag=lag,mean_window_size=mean_window_size, diff_gap_size=diff_gap_size), \
                           get_difference_lag_values(continuous_zero_filled_threshold=continuous_zero_filled_threshold,lag=lag),\
                           get_difference_lag_values(continuous_zero_filled_threshold=continuous_zero_filled_threshold,lag=lag,diff_order = 2), \
                           get_week_of_day(continuous_zero_filled_threshold=continuous_zero_filled_threshold),\
                           get_holiday_for_target(continuous_zero_filled_threshold=continuous_zero_filled_threshold,output_length=output_length), \
                           get_shop_static_info(), \
                           get_target_variables(continuous_zero_filled_threshold=continuous_zero_filled_threshold, output_length=output_length)], True, save_file, True)