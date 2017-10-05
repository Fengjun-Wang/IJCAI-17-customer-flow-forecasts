# -*- coding: utf-8 -*-
import sys
import os
import platform
def getHome():
    """
    return the home directory according to the platform
    :return:
    """
    system = platform.system()
    if system.startswith("Lin"):
        HOME = os.path.expanduser('~')
        sys_name = "Lin"
        HOME_modelFolder = os.path.expanduser('~')
    elif system.startswith("Win"):
        HOME = r"C:\Users\KH44IM"
        sys_name = "Win"
        HOME_modelFolder = r"H:\Model"
    else:
        print "Unknown platform"
        sys_name = "No"
        sys.exit(0)
    return HOME,HOME_modelFolder
sys.path.append(os.path.join(getHome()[0], "Dropbox", "dataset", "Scripts"))
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from tianchi_api.system import getHome
from tianchi_api.metrics import loss
from tianchi_api.system import getHome
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import NearestNeighbors
from tianchi_api.competition import CompetitionPredictionModel,IterativePredictionModel,NewCompetitionPredictionModel
from tianchi_api.features import *
from tianchi_api.getPredictors import predictors_WeatherAirTem
class LL(object):
    def __init__(self, knn_model, train_model, X_population, y_population):
        self.knn_model = knn_model
        self.train_model = train_model
        self.X_population = np.array(X_population)
        self.y_population = np.array(y_population)


    def fit(self):
        self.knn_model.fit(self.X_population)

    def predict(self, X):
        X = np.array(X)
        neighbours = self.knn_model.kneighbors(X, return_distance=False)
        res = []
        for i, row in enumerate(neighbours):
            X_neighbours = self.X_population[row]
            y_neighbours = self.y_population[row]
            #print "training local model..."
            self.train_model.fit(X_neighbours, y_neighbours)
            cur_row_pred = self.train_model.predict(X[[i]])
            res.append(cur_row_pred)
        res = np.vstack(res)
        if res.shape[1]==1:
            res = res.ravel()
        return res



def load_data_set(continuous_zero_filled_threshold,consider_anomaly,lag,outputlength,startshop,endshop):
    HOME = getHome()
    SourceFolder = os.path.join(HOME, \
                                "Dropbox", "dataset", "Analysis",\
                                "PayTrend_Filled_threshold_%s_%s_anomaly"%(continuous_zero_filled_threshold, "consider" if consider_anomaly else "not_consider"), \
                                "NewFeatures")
                                #"training_allDefaultFeatures_lag%s_output%s"%(lag,outputlength))
    list_df = []
    print "loading data"
    for shop_id in xrange(startshop, endshop+1):
        if shop_id%200==0:
            print shop_id
        src_csv_file = "feature_shop_%s.csv" % shop_id
        src_csv_file = os.path.join(SourceFolder, src_csv_file)
        each_shop = pd.read_csv(src_csv_file)
        if each_shop.shape[0]!=0:
            list_df.append(each_shop)
    list_df = pd.concat(list_df, ignore_index=True)
    print 'finish loading'
    return list_df

def test():
    n_neighbors = 10000
    leaf_size = 500
    neigh = NearestNeighbors(n_neighbors=n_neighbors, leaf_size=leaf_size, algorithm='auto')

    n_estimators = 300
    min_samples_split = 2
    min_samples_leaf = 1

    continuous_zero_filled_threshold = 5
    consider_anomaly = True
    lag = 21
    output_length = 14
    startshop = 1
    endshop = 2000
    estimator = ExtraTreesRegressor(n_estimators=n_estimators, max_features="auto", min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf, n_jobs=-1)

    source = load_data_set(continuous_zero_filled_threshold, consider_anomaly, lag, output_length, startshop, endshop)

    estimator_out_length = 7
    all_targets = ['Tar_' + str(i) for i in xrange(1, output_length + 1)]
    pred_targets = all_targets[:estimator_out_length]
    removePreds = predictors_WeatherAirTem(range(1,estimator_out_length+1))
    predictors = [i for i in source.columns.values if i not in (['shop_id', 'day'] + all_targets+removePreds)]

    X_population = source[predictors]
    y_population = source[pred_targets]
    if len(pred_targets) == 1:
        y_population = y_population.values.ravel()

    local_learn_model = LL(neigh, estimator, X_population, y_population)
    print "fitting neighbours..."
    local_learn_model.fit()
    print "finish fitting"

    # mean_window_size = 3
    # std_window_size = 3
    # diff_gap_size = 3
    how_to_deal_with_remaining_zeros='filled'
    fgfs = [get_lag_days_for_competition(),
            get_mean_lag_values_for_competition(),
            get_hmean_lag_values_for_competition(),
            get_std_lag_values_for_competition(),
            get_median_lag_values_for_competition(),
            get_mean_for_alllag_days_for_competition(),
            get_hmean_for_alllag_days_for_competition(),
            get_std_for_alllag_days_for_competition(),
            get_median_for_alllag_days_for_competition(),
            get_skewness_for_alllag_days_for_competition(),
            get_kurtosis_for_alllag_days_for_competition(),
            get_mean_diff_lag_values_for_competition(),
            get_difference_lag_values_for_competition(),
            get_difference_lag_values_for_competition(diff_order=2),
            get_ratio_lag_values_for_competition(),
            get_ratio_lag_values_for_competition(ratio_order=2),
            get_month_of_year_for_competition(),
            get_day_of_month_for_competition(),
            get_week_of_day_for_competition(),
            get_holiday_for_target_for_competition(),
            get_weather_aqi_for_target_for_competition(),
            get_shop_static_info_for_competition()]
    # fgfs = [get_lag_days_for_competition(lag=lag),
    #         get_mean_lag_values_for_competition(lag=lag, mean_window_size=mean_window_size),
    #         get_std_lag_values_for_competition(lag=lag, std_window_size=std_window_size),
    #         get_mean_diff_lag_values_for_competition(lag=lag, mean_window_size=mean_window_size, diff_gap_size=diff_gap_size),
    #         get_difference_lag_values_for_competition(lag=lag),
    #         get_difference_lag_values_for_competition(lag=lag, diff_order = 2),
    #         get_week_of_day_for_competition(),
    #         get_holiday_for_target_for_competition(output_length=output_length),
    #         get_shop_static_info_for_competition()]
    competition_result_file = "1901_2000anew_features_neighbours_%s_alg_%s_estimator_%s_mss_%s_msl_%s_iter_%sx%s_%s.csv"%(n_neighbors,"ETR",n_estimators,min_samples_split,min_samples_leaf,estimator_out_length,14/estimator_out_length,how_to_deal_with_remaining_zeros)

    cpm = NewCompetitionPredictionModel(local_learn_model, estimator_out_length, 7, fgfs, predictors, continuous_zero_filled_threshold, consider_anomaly, how_to_deal_with_remaining_zeros)
    cpm.do_competition(save_to_file=competition_result_file,shoprange=range(1901,2001))


if __name__ == '__main__':
    test()
