# -*- coding: utf-8 -*-
'''
The improved KNN predictor considers all the features.
'''

import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import sys
import os
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor

HOME = os.path.expanduser('~')
import sys

sys.path.append(os.path.join(HOME, "Dropbox", "dataset", "Scripts"))
from GenFeatures import getFeature
import datetime
from GenFeatures import gen_fea_with_zero_multiOutput
from tianchi_api.metrics import loss

WEIGHTS = {'p_day_weight': [1],
           'P_day_mean_weight': [1],
           'p_day_diff_weight': [30,100],
           'week_of_day_weight': [300,600],
           'holiday_weight': [300,600],
           'shopinfo_perpay_weight': [30,100],
           'shopinfo_score_weight': [30,100],
           'shopinfo_comment_weight': [30,100],
           'shopinfo_level_weight': [30,100],
           'shopinfo_city_weight': [30,100],
           'shopinfo_cate_weight': [30,100]
           }


def get_shop_static_info():
    FILE = os.path.join(HOME, "Dropbox", "dataset", "Analysis", "Features", "shop_info_feature.csv")
    return pd.read_csv(FILE)


def get_feature_including_mean_diff(shop_id, lag, outputlength):
    """
    :param shop_id:
    :param lag: number of previous days
    :param outputlength: length of target
    :param val_ratio: ratio of validation set
    :param horizon: the length of the days you want to predict
    :return: the feature data frame
    """
    DesFolder = os.path.join(HOME, "Dropbox", "dataset", "Analysis", "PayTrend_Filled_Mean_timeSeries",
                             "P_filled_%stimeSeries_%s" % (lag, outputlength))
    des_csv_file = "feature_shop_%s.csv" % shop_id
    des_csv_file = os.path.join(DesFolder, des_csv_file)
    if os.path.exists(des_csv_file):
        return pd.read_csv(des_csv_file, parse_dates=['day'])
    else:
        res = gen_fea_with_zero_multiOutput(lag, shop_id, False, outputlength, True, True, True)
        try:
            os.makedirs(DesFolder)
        except:
            pass
        res.to_csv(des_csv_file, index=False)
        return res


def get_truth_value_last_elements(shop_id, lag, outputlength, horizon):
    res = get_feature_including_mean_diff(shop_id, lag, outputlength)
    res = res.iloc[res.shape[0] - horizon:, :]
    return res[['Tar_' + str(i) for i in range(1, outputlength + 1)]]


def get_feature_including_mean_diff_shopStaticInfo(shop_id, lag, outputlength, weights):
    '''
    :param shop_id:
    :param lag: number of previous days
    :param outputlength: length of target
    :return:
    '''
    shop_dynamic_info = get_feature_including_mean_diff(shop_id, lag, outputlength)
    shop_static_info = get_shop_static_info()
    res = pd.merge(shop_dynamic_info, shop_static_info, how='left', on=['shop_id'])
    shopid = res['shop_id']
    days = res['day']
    res = res.iloc[:, np.logical_and(res.columns != 'shop_id', res.columns != 'day')]
    weight = [weights['p_day_weight']] * lag
    #print [weights['P_day_mean_weight']]
    weight += [weights['P_day_mean_weight']] * lag
    weight += [weights['p_day_diff_weight']] * lag
    weight += [weights['week_of_day_weight']] * 7
    weight += [weights['holiday_weight']] * 1
    weight += [1.0] * outputlength  ##Target values
    weight += [weights['shopinfo_perpay_weight']]
    weight += [weights['shopinfo_score_weight']]
    weight += [weights['shopinfo_comment_weight']]
    weight += [weights['shopinfo_level_weight']]
    weight += [weights['shopinfo_city_weight']] * 21
    weight += [weights['shopinfo_cate_weight']] * 73

    weightColumns = ['P_' + str(i) for i in range(lag, 0, -1)]
    weightColumns += ['P_%s_mean_%s' % (i, 7) for i in range(lag, 0, -1)]
    weightColumns += ['P_%s_mean_%s_diff_%s' % (i, 7, 7) for i in range(lag, 0, -1)]
    weightColumns += ['dayOfWeek_%s' % i for i in range(7)]
    weightColumns += ['Holiday_%s' % i for i in range(1, 1 + outputlength)]
    weightColumns += ['Tar_%s' % i for i in range(1, 1 + outputlength)]
    weightColumns += ['per_pay', 'score', 'comment_cnt', 'shop_level', '上海',
                      '杭州', '北京', '广州', '南京', '武汉', '深圳', '温州', '苏州', '宁波', '福州',
                      '成都', '厦门', '绍兴', '无锡', '济南', '金华', '青岛', '合肥', '常州', '其他城市',
                      '美食', '超市便利店', '休闲娱乐', '医疗健康', '美发/美容/美甲', '购物', '休闲茶饮', '超市',
                      '休闲食品', '烘焙糕点', '快餐', '小吃', '中餐', '火锅', '汤/粥/煲/砂锅/炖菜', '便利店',
                      '其他美食', '网吧网咖', '烧烤', '药店', '美容美发', '本地购物', '个人护理', '饮品/甜点',
                      '超市.1', '奶茶', '生鲜水果', '面包', '西式快餐', '其它小吃', '东北菜', '中式快餐',
                      '麻辣烫/串串香', '粥', '蛋糕', '便利店.1', '西餐', '米粉/米线', '川味/重庆火锅', '川菜',
                      '面点', '冰激凌', '网吧网咖.1', '其它快餐', '咖啡厅', '粤菜', '其它烘焙糕点', '中式烧烤',
                      '江浙菜', '零食', '砂锅/煲类/炖菜', '日韩料理', '西北菜', '其它地方菜', '其它休闲食品',
                      '药店.1', '海鲜', '咖啡', '其它火锅', '其他餐饮美食', '湖北菜', '自助餐', '美食特产',
                      '美容美发.1', '香锅/烤鱼', '台湾菜', '闽菜', '湘菜', '熟食', '其它烧烤', '上海本帮菜',
                      '本地购物.1', '个人护理.1']

    weight = pd.Series(weight)
    weight.index = weightColumns
    res = res.multiply(weight)
    res['shop_id'] = shopid
    res['day'] = days
    return res


def getDataSet_Tr_Va(shop_id, lag, outputlength, val_ratio, horizon, weights):
    '''
    :param shop_id: indicate the shop
    :param lag: the feature lag
    :param outputlength: target length
    :param horizon_ the length we wanted to predict
    :return: the train, validation set
    '''
    featuredf = get_feature_including_mean_diff_shopStaticInfo(shop_id, lag, outputlength, weights)
    featuredf = featuredf.dropna()
    train_val = featuredf.iloc[:featuredf.shape[0] - horizon, :]
    X_train, X_test, y_train, y_test = train_test_split(
        train_val, [0] * train_val.shape[0], test_size=val_ratio)
    Xtrain = X_train.iloc[:, X_train.columns != 'Tar_1']
    Ytrain = X_train['Tar_1']
    Xval = X_test.iloc[:, X_test.columns != 'Tar_1']
    Yval = X_test['Tar_1']
    return Xtrain, Ytrain, Xval, Yval


def getTest_last_horizon(shop_id, lag, outputlength, horizon):
    """
    :param shop_id:
    :param lag:
    :param outputlength:
    :param horizon: indicate the last elements we want to predict
    :return:
    """
    featuredf = get_feature_including_mean_diff_shopStaticInfo(shop_id, lag, outputlength)
    featuredf.dropna()
    test = featuredf.iloc[featuredf.shape[0] - horizon:, :]
    Xtest = test.iloc[:, test.columns != 'Tar_1']
    Ytest = test['Tar_1']
    return Xtest, Ytest


def getDataSet_Tr_Va_AllShops(lag, outputlength, val_ratio, horizon, weights):
    shopIds = range(1, 2001)
    Xtrains = []
    Ytrains = []
    Xvals = []
    Yvals = []
    for i in shopIds:
        xt, yt, xv, yv = getDataSet_Tr_Va(i, lag, outputlength, val_ratio, horizon, weights)
        Xtrains.append(xt)
        Ytrains.append(yt)
        Xvals.append(xv)
        Yvals.append(yv)
    Xtrains = pd.concat(Xtrains)
    Ytrains = pd.concat(Ytrains)
    Xvals = pd.concat(Xvals)
    Yvals = pd.concat(Yvals)
    return Xtrains, Ytrains, Xvals, Yvals


def getRegressor(n_neighbors, Average_weights, lag, outputlength, train_ratio, horizon, weights):
    Xt, yt, Xv, yv = getDataSet_Tr_Va_AllShops(lag, outputlength, 1.0 - train_ratio, horizon, weights)
    Xt = Xt.iloc[:, np.logical_and(Xt.columns != 'shop_id', Xt.columns != 'day')]
    # print "Xt shape:",Xt.shape
    neigh = KNeighborsRegressor(n_neighbors=n_neighbors, weights=Average_weights)
    neigh.fit(Xt, yt)
    return neigh


def getMean(x, index, mean_length):
    return np.mean(x[index - mean_length + 1:index + 1])


def getDiff(x, index, mean_length, diff_gap):
    return getMean(x, index, mean_length) - getMean(x, index - diff_gap, mean_length)


def getFeatureVec(X, index, mean_length, diff_gap, lag, weights):
    """
    :param X: a vector with enough length
    :param index: starting point for generating feature
    :param mean_length: length to calculate mean
    :param diff_gap: gap between two means
    :return:
    """
    p1 = list(X['P_1'].values)
    vec = p1[index - lag + 1:index + 1]
    for i in range(index - lag + 1, index + 1):
        vec.append(getMean(p1, i, mean_length))
    for i in range(index - lag + 1, index + 1):
        vec.append(getDiff(p1, i, mean_length, diff_gap))
    vec = [X.iloc[index, 0], X.iloc[index, 1]] + vec
    weekofday = [0] * 7
    weekofday[X.iloc[index, 1].weekday()] = 1
    vec.extend(weekofday)
    vec.append(X['Holiday_1'][index])
    weight = []
    weight += [weights['shopinfo_perpay_weight']]
    weight += [weights['shopinfo_score_weight']]
    weight += [weights['shopinfo_comment_weight']]
    weight += [weights['shopinfo_level_weight']]
    weight += [weights['shopinfo_city_weight']] * 21
    weight += [weights['shopinfo_cate_weight']] * 73

    weightColumns = ['per_pay', 'score', 'comment_cnt', 'shop_level', '上海',
                     '杭州', '北京', '广州', '南京', '武汉', '深圳', '温州', '苏州', '宁波', '福州',
                     '成都', '厦门', '绍兴', '无锡', '济南', '金华', '青岛', '合肥', '常州', '其他城市',
                     '美食', '超市便利店', '休闲娱乐', '医疗健康', '美发/美容/美甲', '购物', '休闲茶饮', '超市',
                     '休闲食品', '烘焙糕点', '快餐', '小吃', '中餐', '火锅', '汤/粥/煲/砂锅/炖菜', '便利店',
                     '其他美食', '网吧网咖', '烧烤', '药店', '美容美发', '本地购物', '个人护理', '饮品/甜点',
                     '超市.1', '奶茶', '生鲜水果', '面包', '西式快餐', '其它小吃', '东北菜', '中式快餐',
                     '麻辣烫/串串香', '粥', '蛋糕', '便利店.1', '西餐', '米粉/米线', '川味/重庆火锅', '川菜',
                     '面点', '冰激凌', '网吧网咖.1', '其它快餐', '咖啡厅', '粤菜', '其它烘焙糕点', '中式烧烤',
                     '江浙菜', '零食', '砂锅/煲类/炖菜', '日韩料理', '西北菜', '其它地方菜', '其它休闲食品',
                     '药店.1', '海鲜', '咖啡', '其它火锅', '其他餐饮美食', '湖北菜', '自助餐', '美食特产',
                     '美容美发.1', '香锅/烤鱼', '台湾菜', '闽菜', '湘菜', '熟食', '其它烧烤', '上海本帮菜',
                     '本地购物.1', '个人护理.1']

    weight = pd.Series(weight)
    weight.index = weightColumns
    # res = res.multiply(weights)
    shop_info = get_shop_static_info()
    shop_info = shop_info[shop_info['shop_id'] == vec[0]][weightColumns]
    # shop_info = list(shop_info[0][1:])
    shop_info = shop_info.multiply(weight)
    shop_info = shop_info.values[0]
    vec.extend(shop_info)
    vec = np.array(vec[2:])
    vec = vec.reshape(1, -1)
    return vec


def ItePredict(neight, XMat, num, lag,weights):
    '''

    :param neight: the regressor
    :param X: the feature of first day that needs to be predicted
    :param num: the number of consecutive days to be predicted
    :return: a list of predicted y values with length of num
    '''
    y = []
    for i in range(1, num + 1)[::-1]:
        X = getFeatureVec(XMat, XMat.shape[0] - i, 7, 7, lag,weights)
        # print "X shape",X.shape
        ypred = neight.predict(X)
        y.append(int(round(ypred[0])))
        if i != 1:
            XMat.ix[XMat.shape[0] - i + 1, 'P_1'] = y[-1]
    y = np.array(y)
    y = y.reshape(-1, 1)
    return y


def get_name_from_weights(weights):
    keys = ['p_day_weight',
            'P_day_mean_weight',
            'p_day_diff_weight',
            'week_of_day_weight',
            'holiday_weight',
            'shopinfo_perpay_weight',
            'shopinfo_score_weight',
            'shopinfo_comment_weight',
            'shopinfo_level_weight',
            'shopinfo_city_weight',
            'shopinfo_cate_weight']
    name = ''
    for k in keys:
        name +=  str(weights[k]) + '_'
    return name


def baseline_test_score(n_neighbors, Average_weights, lag, outputlength, train_ratio, horizon, weights):
    shops = range(1, 2001)
    print "Getting regressor..."
    neigh = getRegressor(n_neighbors, Average_weights, lag, outputlength, train_ratio, horizon, weights)
    print "Done"
    yTrue = []
    yPred = []
    res = []
    for i in shops:
        print i
        yt = get_truth_value_last_elements(i, lag, outputlength, horizon)
        yTrue.append(yt)
        yp = ItePredict(neigh, get_feature_including_mean_diff(i, lag, outputlength), horizon, lag,weights)
        if len(yt) != len(yp):
            print "found %s" % i
            sys.exit(0)
        yp = pd.DataFrame(yp)
        yPred.append(yp)
        curloss = loss(yp, yt)
        curloss = [i, curloss]
        res.append(curloss)
    # print yTrue, yPred
    yTrue = pd.concat(yTrue)
    yPred = pd.concat(yPred)
    tmp = loss(yPred, yTrue)
    res.append(['all', tmp])
    DESFOLDER = os.path.join(HOME, "Dropbox", "dataset", "Analysis", "IterativeModel", "Tuning")
    file_name = "neghbours-%s_targetweight-%s_lag-%s_day-%s.csv" % (
    n_neighbors, Average_weights, lag, datetime.datetime.now().date())
    DESFile = os.path.join(DESFOLDER, get_name_from_weights(weights) + file_name)
    final_rec = pd.DataFrame(res, columns=['shop_id', 'loss'])
    final_rec.to_csv(DESFile, index=False)
    return tmp


if __name__ == "__main__":
    # get_feature_including_mean_diff_shopStaticInfo(1,10,1)
    # print baseline_test_score(50,'distance',10,1,1.0,14)
    keys = ['p_day_weight',
            'P_day_mean_weight',
            'p_day_diff_weight',
            'week_of_day_weight',
            'holiday_weight',
            'shopinfo_perpay_weight',
            'shopinfo_score_weight',
            'shopinfo_comment_weight',
            'shopinfo_level_weight',
            'shopinfo_city_weight',
            'shopinfo_cate_weight']
    weight = {}
    for p_day in WEIGHTS[keys[0]]:
        weight[keys[0]] = p_day
        for p_day_mean in WEIGHTS[keys[1]]:
            weight[keys[1]] = p_day_mean
            for p_day_mean_diff in WEIGHTS[keys[2]]:
                weight[keys[2]] = p_day_mean_diff
                for weekday in WEIGHTS[keys[3]]:
                    weight[keys[3]] = weekday
                    for holi in WEIGHTS[keys[4]]:
                        weight[keys[4]] = holi
                        for shop_perpay in WEIGHTS[keys[5]]:
                            weight[keys[5]] = shop_perpay
                            for shop_score in WEIGHTS[keys[6]]:
                                weight[keys[6]] = shop_score
                                for shop_com in WEIGHTS[keys[7]]:
                                    weight[keys[7]] = shop_com
                                    for shop_lev in WEIGHTS[keys[8]]:
                                        weight[keys[8]] = shop_lev
                                        for shop_city in WEIGHTS[keys[9]]:
                                            weight[keys[9]] = shop_city
                                            for shop_cate in WEIGHTS[keys[10]]:
                                                weight[keys[10]] = shop_cate
                                                print weight
                                                baseline_test_score(50, 'distance', 10, 1, 1.0, 14, weight)
