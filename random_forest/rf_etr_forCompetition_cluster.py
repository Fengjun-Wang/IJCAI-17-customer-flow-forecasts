# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import time
import operator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import ExtraTreesRegressor as ETR
import platform
import sys
import pickle
import xgboost as xgb
from xgboost.sklearn import XGBRegressor

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
        HOME = r"C:\Users\SI30YD"
        sys_name = "Win"
        HOME_modelFolder = r"H:\Model"
    else:
        print "Unknown platform"
        sys_name = "No"
        sys.exit(0)
    return HOME,HOME_modelFolder
HOME, HOME_modelFolder= getHome()
sys.path.append(os.path.join(HOME, "Dropbox", "dataset", "Scripts"))
from tianchi_api.system import getHome
from tianchi_api.metrics import loss, loss_reverse
from sklearn.model_selection import KFold
from tianchi_api.metrics import loss
from tianchi_api.competition import CompetitionPredictionModel, NewCompetitionPredictionModel
# , IterativePredictionModel
from tianchi_api.features import *

def Model_for_competition(algorithm_name, pickle_model_file_name, fgfs, ReportFolder, source, shop_range, predictors_type, predictors, target_variables, estimator_output_length = 14, required_prediction_length = 14, ifSaveModel = False, predict_mode = ["drop", "filled"], n_estimators = 300, min_samples_split =2, min_samples_leaf =1, iterOrCopy = "iterative", estimator_withParams = "l", xgb_param = "l", early_stopping_rounds = 50):

    target_variables_full = target_variables
    target_variables = target_variables[0:estimator_output_length]

    tmp_predictors = ['shop_id', 'day'] + predictors
    X = source[tmp_predictors]
    target_variables_plus = target_variables_full + ['shop_id', 'day']
    y = source[target_variables_plus]



    if 2 < 1:
        estimator_withParams = pickle.load(open(pickle_model_file_name,'rb'))
    else:
        X_train = X[predictors]
        y_train = y[target_variables]
        if len(target_variables)==1:
            y_train=y_train.values.reshape(-1,)
        estimator_withParams.fit(X_train, y_train)
        if not os.path.exists(pickle_model_file_name) and ifSaveModel == True:
            pickle.dump(estimator_withParams,open(pickle_model_file_name,'wb'))


    for mode in predict_mode:
        competition_result_file = "%s_comp_%s_%s_%s_iterLength_%smode_%s_tarLength_%s_iterOrC_%s.csv" % (
            algorithm_name, n_estimators, min_samples_split, min_samples_leaf, estimator_output_length, mode,
            required_prediction_length, iterOrCopy)
        competition_result = os.path.join(ReportFolder, competition_result_file )
        x = NewCompetitionPredictionModel(estimator_withParams, estimator_output_length, required_prediction_length, fgfs, predictors, 5, True, mode)
        # x.predict_shop(shop_id = 1)
        x.do_competition(save_to_file = competition_result, shoprange = shop_range)

if __name__=='__main__':
    list_df = []
    ReportFolder = os.path.join(HOME, "Dropbox", "dataset", "Analysis", "competition", "cluster")
    ModelFolder = os.path.join(HOME_modelFolder, "TianChiModel")
    try:
        os.makedirs(ReportFolder)
    except:
        pass

    continuous_zero_filled_threshold = 5
    consider_anomaly = True
    lag = 21
    source_tar_length = 14
    ave_window_size = 7
    diff_window_size = 7
    SourceFolder = os.path.join(HOME, "Dropbox", "dataset", "Analysis", "PayTrend_Filled_threshold_%s_%s_anomaly" % (
        continuous_zero_filled_threshold, "consider" if consider_anomaly else "not_consider"),
                                "training_allDefaultFeatures_%s_%s"%(lag,source_tar_length))
    fgfs = [get_lag_days_for_competition(), get_mean_lag_values_for_competition(),
            get_std_lag_values_for_competition(), get_mean_diff_lag_values_for_competition(),
            get_difference_lag_values_for_competition(), get_difference_lag_values_for_competition(diff_order=2),
            get_week_of_day_for_competition(),
            get_holiday_for_target_for_competition(), get_shop_static_info_for_competition()]

    cluster_pickle_file = os.path.join(HOME, "Dropbox/dataset/Analysis/Features/Cluster_shops/norm_kmeans/round0",
                                       "6.pickle")
    cluster_result = pickle.load(open(cluster_pickle_file, 'r'))
    getPeriod = -50


    for algorithm_name in ["etr"]:
        #     ["xgBoost", "rf", "etr"]
        for estimator_output_length in [14, 6]:
            iterOrCopy = "iterative"
            required_prediction_length = 14
            if algorithm_name != "xgBoost":
                for n_estimators in [500]:
                    for min_samples_split in [2]:
                     for min_samples_leaf in [1]:
                            if algorithm_name == "rf":
                                            estimator_withParams = RFR(n_estimators=n_estimators, max_features="auto", min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, oob_score=False, n_jobs=-1, random_state = 2017)
                            if algorithm_name == "etr":
                                            estimator_withParams = ETR(n_estimators=n_estimators, max_features="auto",
                                                           min_samples_split=min_samples_split,
                                                           min_samples_leaf=min_samples_leaf, oob_score=False, n_jobs=-1,random_state = 2017)
                            # for key in cluster_result.keys():
                            for key in [0,1,2,3,4,5]:
                                cur_shop_cluster = cluster_result[key]

                                print "loading training dataset from cluster_%s" % key
                                for shop_id in cur_shop_cluster:
                                    src_csv_file = "feature_shop_%s.csv" % shop_id
                                    src_csv_file = os.path.join(SourceFolder, src_csv_file)
                                    each_shop = pd.read_csv(src_csv_file)
                                    each_shop.drop(each_shop.index[:getPeriod], inplace=True)
                                    if each_shop.shape[0] != 0:
                                        list_df.append(each_shop)
                                source = pd.concat(list_df, ignore_index=True)
                                print "finish loading cluster_%s!" % key
                                target_variables = []
                                for i in xrange(1, source_tar_length + 1):
                                    target_variables.append('Tar_' + str(i))
                                all_vars = list(source.columns.values)
                                # for predictors_type in ['withShopStatic']:
                                predictors_type = 'withShopStatic'
                                remove_list = ['day', 'shop_id']
                                predictors = [i for i in all_vars if
                                              ((i not in target_variables) and (i not in remove_list))]
                                Model_for_competition(algorithm_name = algorithm_name, estimator_withParams = estimator_withParams, pickle_model_file_name = "1.csv", fgfs = fgfs, ReportFolder = ReportFolder, source = source, predictors_type = predictors_type, predictors = predictors,target_variables = target_variables, estimator_output_length=estimator_output_length, required_prediction_length=required_prediction_length, ifSaveModel=False, predict_mode=["filled"],  n_estimators = n_estimators, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf,iterOrCopy = iterOrCopy, shop_range=cur_shop_cluster)


