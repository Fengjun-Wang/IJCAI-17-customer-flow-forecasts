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
from tianchi_api.competition import CompetitionPredictionModel, IterativePredictionModel, NewCompetitionPredictionModel
from tianchi_api.features import *
from tianchi_api.models import *

def Model_for_competition(algorithm_name, pickle_model_file_name, fgfs, ReportFolder, source, predictors_type, shop_range, key, predictors, target_variables, estimator_output_length = 14, required_prediction_length = 14, ifSaveModel = False, predict_mode = ["drop", "filled"], n_estimators = 300, min_samples_split =2, min_samples_leaf =1, iterOrCopy = "iterative", estimator_withParams = "l", xgb_param = "l", early_stopping_rounds = 50):

    target_variables_full = target_variables
    target_variables = target_variables[0:estimator_output_length]

    tmp_predictors = ['shop_id', 'day'] + predictors
    X = source[tmp_predictors]
    target_variables_plus = target_variables_full + ['shop_id', 'day']
    y = source[target_variables_plus]


    if 2 < 1:
        estimator_withParams = pickle.load(open(pickle_model_file_name,'rb'))
    else:
        # X_train_forRounds, X_test_forRounds, y_train_forRounds, y_test_forRounds = train_test_split(X, y, test_size=0.1, random_state=0)
        # X_train_forRounds = X_train_forRounds[predictors]
        # y_train_forRounds = y_train_forRounds[target_variables]
        # X_test_forRounds = X_test_forRounds[predictors]
        # y_test_forRounds = y_test_forRounds[target_variables]
        X_train_forRounds = X[predictors][:-1]
        y_train_forRounds = y[:][:-1]
        X_test_forRounds = X[predictors][-1:]
        y_test_forRounds = y[:][-1:]

        X_train = X[predictors]
        y_train = y[target_variables]
        if algorithm_name == "xgBoost":
            xgb_param_list = [{'subsample': 0.9, 'reg_alpha': 100, 'seed': 0, 'colsample_bytree': 0.8,
                         'objective': 'reg:linear','learning_rate': 0.04, 'max_depth': 6, 'min_child_weight': 1, 'gamma': 0.0}]
            num_list = [3500]
            clf_list = []
            # for day in xrange(1, estimator_output_length+1):
            for day in xrange(1,8):
                tar = 'Tar_%s'%day
                # xgtrain = xgb.DMatrix(source[predictors].values, label=source[tar].values,
                #                       feature_names=X_train.columns.values)
                # cvresult = xgb.cv(xgb_param_list[0], xgtrain, num_boost_round=5000, nfold=3,
                #                   early_stopping_rounds=50, seed=0, show_stdv=False)
                #
                # rounds_nice = cvresult.shape[0]
                #
                # with open('num_rounds.csv', 'a+') as fw:
                #     fw.write(str(key) + ' ')
                #     fw.write(str(tar) + '\:')
                #     fw.write(str(rounds_nice) + '\n')
                y_train_tmp = y_train_forRounds[[tar]]
                y_test_tmp = y_test_forRounds[[tar]]
                dtrain = xgb.DMatrix(X_train_forRounds.values, y_train_tmp[tar].values, feature_names=X_train_forRounds.columns.values)
                deval = xgb.DMatrix(X_test_forRounds.values, y_test_tmp[tar].values, feature_names=X_test_forRounds.columns.values)
                watchlist = [(dtrain, 'train'), (deval, 'val')]
                clf = xgb.train(xgb_param_list[0], dtrain, num_boost_round=5000,
                                early_stopping_rounds=100, evals=watchlist)
                # clf = xgb.train(xgb_param_list[0], dtrain, num_boost_round=cvresult.shape[0],
                #                 early_stopping_rounds=50, evals=watchlist)

                y_pred = clf.predict(xgb.DMatrix(X_test_forRounds.values, feature_names=X_test_forRounds.columns.values))
                loss_score = loss(y_pred, y_test_tmp[tar], ifchecked=False)
                # with open(, 'a+') as fw:
                #     fw.write(str(key) + ' ')
                #     fw.write(str(tar) + ' ')
                #     fw.write(str(loss_score))
                #     fw.write('\n')

                all_shop_ids = set(y_test_tmp['shop_id'])
                loss_score_shop_list = []
                for sh in all_shop_ids:
                    row = [sh]
                    row.append(loss(y_pred[np.array(y_test_tmp['shop_id'] == sh), :],y_test_tmp[y_test_tmp['shop_id'] == sh].iloc[:,np.logical_and(y_test_tmp.columns != 'shop_id', y_test_tmp.columns != 'day')]))
                    loss_score_shop_list.append(row)

                results_df = []
                results_df_columns = []
                results_df_columns.extend(['cluster_label', 'Tar_label', 'loss_score','shop_id', 'loss_theLast14Days','bst.best_ntree_limit'])
                for r in loss_score_shop_list:
                    results_df.append(key,day,loss_score, shop_id,r,clf.bst.best_ntree_limit)
                results_df.to_csv('oldFeatures_num_rounds_cluster_shopID_tar.csv', index=False)
            return True

            #     dtrain = xgb.DMatrix(X_train.values, y_train[tar].values, feature_names=X_train.columns.values)
            #     estimator_withParams = xgb.train(xgb_param_list[0], dtrain, num_boost_round= rounds_nice)
            #     estimator_withParams = XGB_Wraper(estimator_withParams)
            #     clf_list.append(estimator_withParams)
            # clf_total = Multiout_Model(*clf_list)
            # for iterOrCopy in ["iterative"]:
            #     for mode in predict_mode:
            #         competition_result_file = "%s_comp_iterLength_%smode_%s_tarLength_%s_iterOrC_%s.csv" % (
            #         algorithm_name, estimator_output_length, mode,
            #         required_prediction_length, iterOrCopy)
            #         competition_result = os.path.join(ReportFolder, competition_result_file)
            #         x = CompetitionPredictionModel(clf_total, estimator_output_length,
            #                                    required_prediction_length,
            #                                    fgfs, predictors, 5, True, mode)
            #         x.do_competition(save_to_file=competition_result, shoprange = shop_range)

if __name__=='__main__':
    list_df = []
    ReportFolder = os.path.join(HOME, "Dropbox", "dataset", "Analysis", "competition")
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
    all_vars = list(source.columns.values)

    predictors_type = 'withShopStatic'
    remove_list = ['day', 'shop_id']
    predictors = [i for i in all_vars if ((i not in target_variables) and (i not in remove_list))]

    for algorithm_name in ["xgBoost"]:
        #     ["xgBoost", "rf", "etr"]
        for estimator_output_length in [7]:
            required_prediction_length = 14
            for key in cluster_result.keys():
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

                Model_for_competition(algorithm_name = algorithm_name, fgfs = fgfs, ReportFolder = ReportFolder, pickle_model_file_name = '2.csv', source = source, predictors_type = predictors_type, predictors = predictors, target_variables = target_variables, estimator_output_length=estimator_output_length, required_prediction_length=required_prediction_length, ifSaveModel=False, predict_mode=["drop", "filled"], shop_range=cur_shop_cluster, key = key)
                import gc
                gc.collect()
