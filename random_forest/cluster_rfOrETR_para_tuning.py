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

def getHome():
    """
    return the home directory according to the platform
    :return:
    """
    system = platform.system()
    if system.startswith("Lin"):
        HOME = os.path.expanduser('~')
        HOME_modelFolder = os.path.expanduser('~')
    elif system.startswith("Win"):
        HOME = r"C:\Users\SI30YD"
        HOME_modelFolder = r"H:\Model"
    else:
        print "Unknown platform"
        sys.exit(0)
    return HOME, HOME_modelFolder
HOME, HOME_modelFolder = getHome()
sys.path.append(os.path.join(HOME, "Dropbox", "dataset", "Scripts"))
from tianchi_api.system import getHome
from tianchi_api.metrics import loss, loss_reverse
from sklearn.model_selection import KFold
from tianchi_api.metrics import loss
from tianchi_api.competition import CompetitionPredictionModel, NewIterativePredictionModel
from tianchi_api.features import *

def rfOrETR_para_tuning(source, pickle_model_file_name, ReportFolder, estimator_withParams, fgfs, report_file,report_per_shop, predictors_type, predictors, target_variables, iterOrCopy = ["iterative"], estimator_output_length = 14, required_prediction_length = 14, n_folds = 5, ifSavePickle= False, shop_range=range(1,2001)):
    target_variables_full = target_variables
    target_variables = target_variables[0:estimator_output_length]
    tmp_predictors = ['shop_id', 'day'] + predictors
    X = source[tmp_predictors]
    target_variables_plus = ['shop_id', 'day']+target_variables_full
    y = source[target_variables_plus]

    # if os.path.exists(pickle_model_file_name):
    #     print "already trained"
    #     return True

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10,random_state = 0)
    X_train = X[predictors]
    y_train = y[target_variables]
    if len(target_variables)==1:
        y_train = y_train.values.ravel()
    print "fitting regressor.."
    estimator_withParams.fit(X_train, y_train)
    print "finish!"

    # if not os.path.exists(pickle_model_file_name) and ifSavePickle == True:
    #     pickle.dump(estimator_withParams,open(pickle_model_file_name,'wb'))
    feature_imp = {}
    for i in xrange(0, estimator_withParams.feature_importances_.shape[0]):
        feature_imp.setdefault(predictors[i], estimator_withParams.feature_importances_[i])
    sorted_scoreDic = sorted(feature_imp.items(), key=operator.itemgetter(1), reverse=True)

    if algorithm_name == "rf":
        oob_score_result = estimator_withParams.oob_score_

    x = NewIterativePredictionModel(estimator_withParams, estimator_output_length, required_prediction_length, fgfs, predictors,5, True, shop_range)

    for iterOrCopy in ["iterative", "copy"]:
        report_per_shop = "%s_%s_iterLength_%s_n%s_minS%s_minL%s_iterOrCopy_%s.csv" % (
            algorithm_name, predictors_type, iter, n_estimators, min_samples_split, min_samples_leaf, iterOrCopy)
        report_per_shop = os.path.join(ReportFolder, report_per_shop)
        y_pred = x.do_iterative_prediction(X_test[['shop_id', 'day'] ], how = iterOrCopy)
        loss_score = loss(y_pred,y_test[target_variables_full])
        print "loss: ", loss_score

        results_df = []
        results_df_columns = []
        if algorithm_name == "rf":
            results_df.extend([oob_score_result, loss_score, n_estimators, min_samples_split, min_samples_leaf, predictors_type, iter, iterOrCopy])
            results_df_columns.extend(['oob_score_result', 'loss_score', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'predictors_type', 'iterLength', 'iterOrCopy'])

        if algorithm_name == "etr":
            results_df.extend(
                [0, loss_score, n_estimators, min_samples_split, min_samples_leaf, predictors_type, iter,
                 iterOrCopy])
            results_df_columns.extend(
                ['no_oob', 'loss_score', 'n_estimators', 'min_samples_split', 'min_samples_leaf',
                 'predictors_type', 'iterLength', 'iterOrCopy'])

        if iterOrCopy == "iterative":
            for i in xrange(0, len(sorted_scoreDic)):
                results_df_columns.append(sorted_scoreDic[i][0])
                results_df.append(sorted_scoreDic[i][1])
        results_df = pd.DataFrame([results_df], columns = results_df_columns)

        if os.path.exists(report_file):
            results_df.to_csv(report_file, index=False, mode='a')
        else:
            results_df.to_csv(report_file, index=False)
        print "calculating score for each shop!"
        all_shop_ids = set(y_test['shop_id'])
        loss_score_shop_list = []
        for sh in all_shop_ids:
            row=[sh]
            row.append(loss(y_pred[np.array(y_test['shop_id']==sh),:],\
                            y_test[y_test['shop_id']==sh].iloc[:,np.logical_and(y_test.columns != 'shop_id', y_test.columns != 'day')]\
                            )\
                       )
            loss_score_shop_list.append(row)
        # loss_score_dict = {}
        # for i in xrange(0, y_test.shape[0]):
        #     shop_id_test = y_test['shop_id'].values[i]
        #     loss_score_dict.setdefault(shop_id_test, [[], []])[1].extend(
        #         y_test.iloc[i, np.logical_and(y_test.columns != 'shop_id', y_test.columns != 'day')])
        #     loss_score_dict.setdefault(shop_id_test, [[], []])[0].extend(y_pred[i])
        # loss_score_shop_list = []
        # for shop in xrange(1, 2001):
        #     if shop in loss_score_dict:
        #         row = [shop]
        #         this_loss = loss(loss_score_dict[shop][0], loss_score_dict[shop][1])
        #         row.append(this_loss)
        #         loss_score_shop_list.append(row)
        print "finish!"
        with open(report_per_shop, 'a+') as fw:
            for r in loss_score_shop_list:
                    fw.write(','.join(map(str, r)) + '\n')


if __name__=='__main__':
    list_df = []
    ifSavePickle = False

    # algorithm_name = "etr"


    continuous_zero_filled_threshold = 5
    consider_anomaly = True
    lag = 21
    source_tar_length = 14
    ave_window_size = 7
    diff_window_size = 7
    SourceFolder = os.path.join(HOME, "Dropbox", "dataset", "Analysis", "PayTrend_Filled_threshold_%s_%s_anomaly" % (
        continuous_zero_filled_threshold, "consider" if consider_anomaly else "not_consider"),
                                "training_allDefaultFeatures_%s_%s"%(lag,source_tar_length))
    cluster_pickle_file = os.path.join(HOME,"Dropbox/dataset/Analysis/Features/Cluster_shops/norm_kmeans/round0","6.pickle")
    cluster_result = pickle.load(open(cluster_pickle_file,'r'))
    getPeriod = -50
    for key in cluster_result.keys():
        cur_shop_cluster = cluster_result[key]

        print "loading training dataset from cluster_%s"%key
        for shop_id in cur_shop_cluster:
            src_csv_file = "feature_shop_%s.csv" % shop_id
            src_csv_file = os.path.join(SourceFolder, src_csv_file)
            each_shop = pd.read_csv(src_csv_file)
            each_shop.drop(each_shop.index[:getPeriod], inplace=True)
            if each_shop.shape[0] != 0:
                list_df.append(each_shop)
        source = pd.concat(list_df, ignore_index=True)
        print "finish loading cluster_%s!"%key

        target_variables = []
        for i in xrange(1, source_tar_length + 1):
            target_variables.append('Tar_' + str(i))
        all_vars = list(source.columns.values)

        for predictors_type in ['withShopStatic']:
            if predictors_type == 'withoutShopStatic':
                # requires re-writing
                predictors = ['shop_id', 'dayOfWeek_0', 'dayOfWeek_1', 'dayOfWeek_2', 'dayOfWeek_3', 'dayOfWeek_4',
                              'dayOfWeek_5', 'dayOfWeek_6']
                predictors.extend(['P_%s' % i for i in xrange(1, lag + 1)])
                predictors.extend(
                    map(lambda x: "P_" + str(x) + '_mean_' + str(ave_window_size), [i for i in range(1, lag + 1)[::-1]]))
                predictors.extend(
                    map(lambda x: "P_" + str(x) + '_mean_' + str(ave_window_size) + '_diff_' + str(diff_window_size),
                        [i for i in range(1, lag + 1)[::-1]]))
                predictors.extend(['Holiday_%s' % i for i in xrange(1, source_tar_length + 1)])
            if predictors_type == 'withShopStatic':
                remove_list = ['day', 'shop_id']
                predictors = [i for i in all_vars if ((i not in  target_variables  ) and (i not in remove_list))]



            for algorithm_name in ["rf", "etr"]:
                fgfs = [get_lag_days_for_competition(), get_mean_lag_values_for_competition(),
                        get_std_lag_values_for_competition(), get_mean_diff_lag_values_for_competition(),
                        get_difference_lag_values_for_competition(),
                        get_difference_lag_values_for_competition(diff_order=2),
                        get_week_of_day_for_competition(),
                        get_holiday_for_target_for_competition(), get_shop_static_info_for_competition()]
                ReportFolder = os.path.join(HOME, "Dropbox", "dataset", "Analysis", "mimo_rf", "more_features_modified","clusters","cluster_%s"%key,
                                            algorithm_name)
                ModelFolder = os.path.join(HOME_modelFolder, "TianChiModel")
                try:
                    os.makedirs(ReportFolder)
                except:
                    pass


                report_file = "%s_%s_iterLength.csv" % (algorithm_name, predictors_type)
                report_file = os.path.join(ReportFolder, report_file)
                for iter in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
                        for n_estimators in [500]:
                            for min_samples_split in [2]:
                                for min_samples_leaf in [1]:
                                    print "cluster:%s,alg:%s,iter:%s,n_estimator:%s,mss:%s,msl:%s"%(key,algorithm_name,iter,n_estimators,min_samples_split,min_samples_leaf)
                                    pickle_model_file_name = "clu_%s_%s_%s_iterLength_%s_n%s_minS%s_minL%s.csv" % (key,algorithm_name, predictors_type, iter, n_estimators, min_samples_split, min_samples_leaf)
                                    pickle_model_file_name = os.path.join(ModelFolder, pickle_model_file_name)

                                    if algorithm_name == "rf":
                                        estimator_withParams = RFR(n_estimators=n_estimators, max_features="auto", min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, oob_score=True, n_jobs=-1, random_state = 2017)
                                    if algorithm_name == "etr":
                                        estimator_withParams = ETR(n_estimators=n_estimators, max_features="auto",min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, oob_score=False, n_jobs=-1, random_state = 2017)
                                    rfOrETR_para_tuning(source=source, ReportFolder = ReportFolder, pickle_model_file_name = pickle_model_file_name, fgfs = fgfs, estimator_withParams = estimator_withParams, report_file = report_file, report_per_shop = '1.csv', predictors_type = predictors_type, predictors = predictors, target_variables = target_variables, estimator_output_length=iter, required_prediction_length=14, ifSavePickle = ifSavePickle,shop_range=cur_shop_cluster)
