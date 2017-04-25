# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import time
import operator
from sklearn.model_selection import train_test_split, GridSearchCV # for old version, use cross_validation package
from sklearn import metrics
import sys
import platform


def getHome():
    """
    return the home directory according to the platform
    :return:
    """
    system = platform.system()
    if system.startswith("Lin"):
        HOME = os.path.expanduser('~')
    elif system.startswith("Win"):
        HOME = r"C:\Users\KH44IM"
    else:
        print "Unknown platform"
        sys.exit(0)
    return HOME

HOME = getHome()
sys.path.append(os.path.join(HOME, "Dropbox", "dataset", "Scripts"))
from tianchi_api.system import getHome
from tianchi_api.metrics import loss, loss_reverse

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

from sklearn.model_selection import KFold


def scorer(estimator, X, y):
    return loss_reverse(estimator.predict(X),y, False)


def xgBoost_out14(source, day, predictors, predictors_type, ifGS = True,  target_variables = ['Tar_1', 'Tar_2'], ifCompetition = False, useTrainCV=True, cv_folds=5, early_stopping_rounds=50, X_test_comp = 'l'):

    for tar in  target_variables[day-1:day]:
        report_file = "xgBoost_14out_removeSHOPID_%s_day_%s.txt" % (predictors_type, tar)
        report_file = os.path.join(ReportFolder, report_file)

        #1 get data
        X = source[predictors]
        target_variables_plus = [tar]+ ['shop_id', 'day']
        y = source[target_variables_plus]

        # X_train = X[predictors][:-1]
        # y_train = y[:][:-1]
        # X_test = X[predictors][-1:]
        # y_test = y[:][-1:]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)

        # if ifGS:
        #     # initial tuning for default paras -> to set n_estimators
        #     xgb_param = {'reg_alpha': 0, 'subsample': 0.8, 'seed': 0, 'colsample_bytree': 0.8,
        #                  'objective': 'reg:linear', 'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 1,
        #                  'gamma': 0}
        #     dtrain = xgb.DMatrix(X_train.values, y_train[tar].values, feature_names=X_train.columns.values)
        #     deval = xgb.DMatrix(X_test.values, y_test[tar].values, feature_names=X_test.columns.values)
        #     watchlist = [(dtrain, 'train'), (deval, 'val')]
        #     xgtrain = xgb.DMatrix(source[predictors].values, label=source[tar].values,
        #                           feature_names=X_train.columns.values)
        #     cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=10000, nfold=cv_folds,
        #                       early_stopping_rounds=50, seed=0, show_stdv=False)
        #
        #     print cvresult.shape[0]  # so we get the best n_estimators for competition
        #
        #     # clf = xgb.train(xgb_param, dtrain, num_boost_round=cvresult.shape[0], evals=watchlist,
        #     #                 early_stopping_rounds=100)
        #     #
        #     # feature_imp = clf.get_fscore()
        #     # sorted_scoreDic = sorted(feature_imp.items(), key=operator.itemgetter(1), reverse=True)
        #     #
        #     # # report_file_confirm_iter = "xgBoost_14out_removeSHOPID_%s_day_%s.txt" % (predictors_type, tar)
        #     # # report_file_confirm_iter = os.path.join(ReportFolder, report_file_confirm_iter)
        #     #
        #     # y_pred = clf.predict(xgb.DMatrix(X_test.values, feature_names=X_test.columns.values))
        #     # loss_score = loss(y_pred, y_test[tar], ifchecked=False)
        #     # print "loss_score: ", loss_score
        #
        #     with open(report_file, 'a+') as fw:
        #         fw.write("-------------------\Initial set rounds for default paras:\n")
        #         # if tar == 1:
        #         #     fw.write(str(sorted_scoreDic))
        #         fw.write(str(xgb_param))
        #         fw.write("\n")
        #         fw.write(str(cvresult.shape[0]))
        #         # fw.write('loss_score for random 10% samples: ')
        #         # fw.write(str(loss_score))
        #
        #     best_rounds = cvresult.shape[0]
        best_rounds = 1000

        if ifGS:
            #2 Gridsearch to set the best parameters
            cv_params_1 = {'max_depth': [6,7,8,9], 'min_child_weight': [1, 3]}
            ind_params_1 = {'learning_rate': 0.1,  'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                          'objective': 'reg:linear', 'silent':1, 'n_estimators': best_rounds}
            optimized_GBM = GridSearchCV(estimator = xgb.XGBRegressor(**ind_params_1), param_grid = cv_params_1, scoring = scorer, cv=KFold(n_splits=cv_folds, shuffle=True, random_state=0), n_jobs=-1)
            optimized_GBM.fit(X, y[tar])
            print optimized_GBM.cv_results_
            print type(optimized_GBM.cv_results_)
            print optimized_GBM.best_params_, optimized_GBM.best_score_
            best_params_1 = optimized_GBM.best_params_
            best_score_1 = optimized_GBM.best_score_
            with open(report_file, 'a+') as fw:
                fw.write("-------------------!!!!\:\n")
                fw.write(str(optimized_GBM.cv_results_))
                fw.write('\n....:\n' )
                fw.write(str(cv_params_1) + '\n')
                fw.write(str(ind_params_1) + '\n')
                fw.write(str(optimized_GBM.cv_results_['mean_test_score']) + '\n')
                fw.write(str(optimized_GBM.best_params_) + str(optimized_GBM.best_score_) + '\n')

            # gamma: 可选可不选
            cv_params_2 = {'gamma':[0.1, 0.3]}
            ind_params_2 = {'learning_rate': 0.1, 'subsample': 0.8, 'seed': 0, 'colsample_bytree': 0.8,
                          'objective': 'reg:linear', 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': best_rounds}
            ind_params_2.update(best_params_1)
            optimized_GBM = GridSearchCV(estimator=xgb.XGBRegressor(**ind_params_2), param_grid=cv_params_2,
                                         scoring=scorer, cv=KFold(n_splits=cv_folds, shuffle=True, random_state=0),
                                         n_jobs=-1)
            optimized_GBM.fit(X, y[tar])
            print optimized_GBM.cv_results_
            best_params_2 = optimized_GBM.best_params_
            best_score_2 = optimized_GBM.best_score_
            if best_score_2 < best_score_1:
                best_params_2 = {'gamma':0.0}
            best_params_2.update(best_params_1)
            with open(report_file, 'a+') as fw:
                fw.write("-------------------\:\n")
                fw.write(str(optimized_GBM.cv_results_))
                fw.write('\n....:\n')
                fw.write(str(cv_params_2) + '\n')
                fw.write(str(ind_params_2) + '\n')
                fw.write(str(optimized_GBM.cv_results_['mean_test_score']) + '\n')
                fw.write(str(optimized_GBM.best_params_) + str(optimized_GBM.best_score_) + '\n')

            # 3 GridSearch on rest parameters:
            cv_params_3 = { 'subsample': [0.8, 0.9], 'colsample_bytree': [i/10.0 for i in range(8,10)]}
            ind_params_3 = {'learning_rate': 0.1, 'seed': 0, 'colsample_bytree': 0.8,
                          'objective': 'reg:linear', 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': best_rounds}
            ind_params_3.update(best_params_2)
            optimized_GBM = GridSearchCV(estimator=xgb.XGBRegressor(**ind_params_3), param_grid=cv_params_3,
                                         scoring=scorer, cv=KFold(n_splits=cv_folds, shuffle=True, random_state=0),
                                         n_jobs=-1)
            optimized_GBM.fit(X, y[tar])
            print optimized_GBM.cv_results_
            best_params_3 = optimized_GBM.best_params_
            best_score_3 = optimized_GBM.best_score_
            best_params_3.update(best_params_2)
            with open(report_file, 'a+') as fw:
                fw.write("-------------------\:\n")
                fw.write(str(optimized_GBM.cv_results_))
                fw.write('\n....:\n')
                fw.write(str(cv_params_3) + '\n')
                fw.write(str(ind_params_3) + '\n')
                fw.write(str(optimized_GBM.cv_results_['mean_test_score']) + '\n')
                fw.write(str(optimized_GBM.best_params_) + str(optimized_GBM.best_score_) + '\n')

            cv_params_5 = {'reg_alpha': [ 100, 150]}
            ind_params_5 = {'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'seed': 0,
                            'colsample_bytree': 0.8, 'objective': 'reg:linear', 'max_depth': 6,
                            'min_child_weight': 1, 'n_estimators': best_rounds}
            ind_params_5.update(best_params_3)
            optimized_GBM = GridSearchCV(estimator=xgb.XGBRegressor(**ind_params_5),  param_grid=cv_params_5,
                                         scoring=scorer, cv=KFold(n_splits=cv_folds, shuffle=True, random_state=0),
                                         n_jobs=-1)
            optimized_GBM.fit(X, y[tar])
            print optimized_GBM.cv_results_
            with open(report_file, 'a+') as fw:
                fw.write("-------------------\:\n")
                fw.write(str(optimized_GBM.cv_results_))
                fw.write('\n....:\n')
                fw.write(str(cv_params_5) + '\n')
                fw.write(str(ind_params_5) + '\n')
                fw.write(str(optimized_GBM.cv_results_['mean_test_score']) + '\n')
                fw.write(str(optimized_GBM.best_params_) + str(optimized_GBM.best_score_) + '\n')
            sys.exit(0)

        #3 xgb.cv choose the optimized n_estimators
        if ifGS == False:
            xgb_param = {'reg_alpha': 100, 'colsample_bytree': 0.8, 'learning_rate': 0.05, 'min_child_weight': 3, 'subsample': 0.9, 'seed': 0, 'objective': 'reg:linear', 'max_depth': 7, 'gamma': 0.3}
            # xgb_param = {'reg_alpha': 100, 'subsample': 0.9, 'seed': 0, 'colsample_bytree': 0.7,
            #              'objective': 'reg:linear', 'learning_rate': 0.04, 'max_depth': 7, 'min_child_weight': 1, 'gamma': 0.1}
            dtrain = xgb.DMatrix(X_train.values, y_train[tar].values, feature_names=X_train.columns.values)
            deval = xgb.DMatrix(X_test.values, y_test[tar].values, feature_names=X_test.columns.values)
            watchlist = [(dtrain, 'train'), (deval, 'val')]
            xgtrain = xgb.DMatrix(source[predictors].values, label=source[tar].values,
                                  feature_names=X_train.columns.values)
            cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=10000, nfold=cv_folds,
                              early_stopping_rounds=early_stopping_rounds, seed=0, show_stdv=False)

            print cvresult.shape[0]  # so we get the best n_estimators for competition

            clf = xgb.train(xgb_param, dtrain, num_boost_round=cvresult.shape[0], evals=watchlist,
                            early_stopping_rounds=early_stopping_rounds)

            feature_imp = clf.get_fscore()
            sorted_scoreDic = sorted(feature_imp.items(), key=operator.itemgetter(1), reverse=True)

            # print sorted_scoreDic

            report_file_confirm_iter = "setRounds_xgBoost_14out_removeSHOPID_%s_day_%s.txt" % (predictors_type, tar)
            report_file_confirm_iter = os.path.join(ReportFolder, report_file_confirm_iter)

            y_pred = clf.predict(xgb.DMatrix(X_test.values, feature_names=X_test.columns.values))
            loss_score = loss(y_pred, y_test[tar], ifchecked = False)
            print "loss_score: ", loss_score

            with open(report_file_confirm_iter, 'a+') as fw:
                fw.write("-------------------After choose all paras, test best rounds for a smaller learning rate, 0.04\:\n")
                fw.write(str(sorted_scoreDic))
                fw.write(str(xgb_param))
                fw.write("\n")
                fw.write(str(cvresult.shape[0]))
                fw.write('loss_score: ')
                fw.write(str(loss_score))


if __name__=='__main__':
    ifCompetition = False
    list_df = []
    method = 'allOutput_onetime'
    ReportFolder = os.path.join(HOME, "Dropbox", "dataset", "Analysis", "xgBoost", "perfect_tune","features_final")
    competition_result_file = "competition.csv"
    competition_result_file = os.path.join(ReportFolder, competition_result_file)

    continuous_zero_filled_threshold = 5
    consider_anomaly = True
    SourceFolder = os.path.join(HOME, "Dropbox", "dataset", "Analysis", "PayTrend_Filled_threshold_%s_%s_anomaly" % (
    continuous_zero_filled_threshold, "consider" if consider_anomaly else "not_consider"), "NewFeatures")

    getPeriod = -50

    for shop_id in xrange(1,2001):
        src_csv_file = "feature_shop_%s.csv"%shop_id
        src_csv_file = os.path.join(SourceFolder,src_csv_file)
        each_shop = pd.read_csv(src_csv_file)
        each_shop.drop(each_shop.index[:getPeriod], inplace=True)
        list_df.append(each_shop)
    source = pd.concat(list_df, ignore_index = True)

    lag = 21
    source_tar_length = 14
    ave_window_size = 7
    diff_window_size = 7


    if method == 'allOutput_onetime':
        target_variables = []
        for i in xrange(1,source_tar_length+1):
            target_variables.append('Tar_'+str(i))
        all_vars = list(source.columns.values)

    if ifCompetition:
        DesFolder = os.path.join(HOME, "Dropbox", "dataset", "Analysis", "PayTrend_Filled_Mean_timeSeries",
                             "P_filled_%stimeSeries_%s_competition" % (21, 14), "NewFeatures")
        des_csv_file = "feature_shop_all.csv"
        des_csv_file = os.path.join(DesFolder, des_csv_file)
        X_test_forCom_1 = pd.read_csv(des_csv_file)
        X_test_forCom = pd.merge(X_test_forCom_1, right, how='left', on=['shop_id'])
    else:
        X_test_forCom = '1'

    for para in [1]:
                    print "already read"
                    day = 4
                    remove_list = ['day', 'shop_id']
                    from tianchi_api.getPredictors import predictors_WeatherAirTem
                    remove_list.extend(predictors_WeatherAirTem([day]))
                    predictors = [i for i in all_vars if ((i not in target_variables) and (i not in remove_list))]
                    predictors_type = 'withShopStatic'
                    xgBoost_out14(source, day, predictors, predictors_type = predictors_type, target_variables = target_variables, X_test_comp = X_test_forCom, ifGS=False, ifCompetition=False, useTrainCV=True, cv_folds=5, early_stopping_rounds=50)



