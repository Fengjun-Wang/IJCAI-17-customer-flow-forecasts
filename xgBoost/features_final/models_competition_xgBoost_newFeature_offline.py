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
from tianchi_api.competition import CompetitionPredictionModel, IterativePredictionModel,NewCompetitionPredictionModel
from tianchi_api.features import *
from tianchi_api.models import *

def Model_for_competition(algorithm_name, pickle_model_file_name, fgfs, ReportFolder, source, target_variables, estimator_output_length = 14, required_prediction_length = 14, ifSaveModel = False, predict_mode = ["drop", "filled"], n_estimators = 300, min_samples_split =2, min_samples_leaf =1, iterOrCopy = "iterative", estimator_withParams = "l", xgb_param = "l", early_stopping_rounds = 50):

    all_vars = list(source.columns.values)

    target_variables_full = target_variables
    target_variables = target_variables[0:estimator_output_length]

    predictors_type = 'withShopStatic'
    remove_list = ['day', 'shop_id']
    from tianchi_api.getPredictors import predictors_WeatherAirTem
    # remove_list.extend(predictors_WeatherAirTem([day]))
    predictors = [i for i in all_vars if ((i not in target_variables_full) and (i not in remove_list))]

    tmp_predictors = ['shop_id', 'day'] + predictors
    X = source[tmp_predictors]
    target_variables_plus = target_variables_full + ['shop_id', 'day']
    y = source[target_variables_plus]

    X_train = X[predictors][:-1]
    y_train = y[target_variables][:-1]
    if algorithm_name == "xgBoost":
        xgb_param_list = [{'subsample': 0.9, 'reg_alpha': 100, 'seed': 0, 'colsample_bytree': 0.7, 'gamma': 0.1, 'objective': 'reg:linear', 'learning_rate': 0.04, 'max_depth': 6, 'min_child_weight': 1},
                              {'subsample': 0.9, 'reg_alpha': 100, 'seed': 0, 'colsample_bytree': 0.7, 'gamma': 0.1,
                               'objective': 'reg:linear', 'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 1},
                              {'subsample': 0.9, 'reg_alpha': 150, 'seed': 0, 'colsample_bytree': 0.9, 'gamma': 0.0,
                               'objective': 'reg:linear', 'learning_rate': 0.04, 'max_depth': 6, 'min_child_weight': 1},
                              {'subsample': 0.9, 'reg_alpha': 100, 'seed': 0, 'colsample_bytree': 0.8, 'gamma': 0.3,
                               'objective': 'reg:linear', 'learning_rate': 0.05, 'max_depth': 7, 'min_child_weight': 3},
                              {'subsample': 0.9, 'seed': 0, 'colsample_bytree': 0.8, 'gamma': 0.1,
                               'objective': 'reg:linear', 'learning_rate': 0.05, 'max_depth': 7, 'min_child_weight': 3},
                              {'subsample': 0.9, 'reg_alpha': 150, 'seed': 0, 'colsample_bytree': 0.8, 'gamma': 0.0,
                               'objective': 'reg:linear', 'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 1},
                              {'subsample': 0.9, 'reg_alpha': 1, 'seed': 0, 'colsample_bytree': 0.7,
                               'min_child_weight': 1, 'objective': 'reg:linear', 'learning_rate': 0.05, 'max_depth': 7,
                               'gamma': 0.1}
                              ]

        num_list = [3229,3338,5419,3204,3837,4265,4882]
        # num_list = [10,10,10,10,10,10,10]
        clf_list = []
        predictors_list = []
        for day in xrange(1, estimator_output_length+1):
            tar = 'Tar_%s'%day

            remove_list = ['day', 'shop_id']
            remove_list.extend(predictors_WeatherAirTem([day]))
            predictors = [i for i in all_vars if ((i not in target_variables_full) and (i not in remove_list))]
            predictors_list.append(predictors)

            dtrain = xgb.DMatrix(X_train[predictors].values, y_train[tar].values, feature_names=X_train[predictors].columns.values)
            estimator_withParams = xgb.train(xgb_param_list[day-1], dtrain, num_boost_round= num_list[day-1])
            estimator_withParams = XGB_Wraper(estimator_withParams)
            clf_list.append(estimator_withParams)
        clf_total = Multiout_Model(clf_list,predictors_list)
        for iterOrCopy in ["iterative", "copy"]:
            for mode in predict_mode:
                save_to_file_shop_loss = "%s_shop_iterLength_%smode_%s_tarLength_%s_iterOrC_%s.csv" % (
                algorithm_name, estimator_output_length, mode,
                required_prediction_length, iterOrCopy)
                save_to_file_shop_loss = os.path.join(ReportFolder, save_to_file_shop_loss)

                save_to_file_shop_day_loss = "%s_shop_day_iterLength_%smode_%s_tarLength_%s_iterOrC_%s.csv" % (
                    algorithm_name, estimator_output_length, mode,
                    required_prediction_length, iterOrCopy)
                save_to_file_shop_day_loss = os.path.join(ReportFolder, save_to_file_shop_day_loss)

                save_to_file_day_loss = "%s_day_iterLength_%smode_%s_tarLength_%s_iterOrC_%s.csv" % (
                    algorithm_name, estimator_output_length, mode,
                    required_prediction_length, iterOrCopy)
                save_to_file_day_loss = os.path.join(ReportFolder, save_to_file_day_loss)
                x = NewCompetitionPredictionModel(clf_total, estimator_output_length,
                                               required_prediction_length,
                                               fgfs, None, 5, True, mode)
                x.do_last14_test(how = iterOrCopy, save_to_file_shop_loss= save_to_file_shop_loss, save_to_file_day_loss= save_to_file_day_loss, save_to_file_shop_day_loss=save_to_file_shop_day_loss, shoprange= range(1,2001))
                # x.predict_shop(shop_id = 1)
                # x.do_competition(save_to_file=competition_result, shoprange = range(1,2001), how = iterOrCopy)




if __name__=='__main__':
    list_df = []
    ReportFolder = os.path.join(HOME, "Dropbox", "dataset", "Analysis", "competition", "features_final")
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
                                 "NewFeatures")
    # fgfs = [get_lag_days_for_competition(), get_mean_lag_values_for_competition(),
    #         get_std_lag_values_for_competition(), get_mean_diff_lag_values_for_competition(),
    #         get_difference_lag_values_for_competition(), get_difference_lag_values_for_competition(diff_order=2),
    #         get_week_of_day_for_competition(),
    #         get_holiday_for_target_for_competition(), get_shop_static_info_for_competition()]
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

    getPeriod = -50

    for shop_id in xrange(1, 2001):
        src_csv_file = "feature_shop_%s.csv" % shop_id
        src_csv_file = os.path.join(SourceFolder, src_csv_file)
        each_shop = pd.read_csv(src_csv_file)
        each_shop.drop(each_shop.index[:getPeriod], inplace=True)
        if each_shop.shape[0]!=0:
            list_df.append(each_shop)
    source = pd.concat(list_df, ignore_index=True)

    target_variables = []
    for i in xrange(1, source_tar_length + 1):
        target_variables.append('Tar_' + str(i))



    # for predictors_type in ['withShopStatic']:


    for algorithm_name in ["xgBoost"]:
        #     ["xgBoost", "rf", "etr"]
        for estimator_output_length in [7]:
            required_prediction_length = 14
            if algorithm_name == "xgBoost":
                Model_for_competition(algorithm_name = algorithm_name, fgfs = fgfs, ReportFolder = ReportFolder, pickle_model_file_name = '2.csv', source = source, target_variables = target_variables, estimator_output_length=estimator_output_length, required_prediction_length=required_prediction_length, ifSaveModel=False, predict_mode=["filled"])
                import gc
                gc.collect()