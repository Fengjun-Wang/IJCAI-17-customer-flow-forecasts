# -*- coding: utf-8 -*-
'''
The baseline KNN predictor only considers the previous seven days.
No other features are considered like shop_info, weather.
No parameters are tuned like K, distance weight, average weight.
This is only for first submitting and intuitive experience of what this method can get us.
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


def getDataSet_Tr_Va(shop_id, lag, is_nozero, val_ratio, horizon):
    '''
    :param shop_id: indicate the shop
    :param lag: the feature lag
    :param is_nozero: with or without zero points
    :return: the train, validation set
    '''
    featuredf = getFeature(shop_id, lag, is_nozero)
    train_val = featuredf.iloc[:featuredf.shape[0] - horizon, :]
    X_train, X_test, y_train, y_test = train_test_split(
        train_val, [0] * train_val.shape[0], test_size = val_ratio)
    Xtrain = X_train.iloc[:, :X_train.shape[1] - 1]
    Ytrain = X_train.iloc[:, X_train.shape[1] - 1]
    Xval = X_test.iloc[:, :X_test.shape[1] - 1]
    Yval = X_test.iloc[:, X_test.shape[1] - 1]
    return Xtrain, Ytrain, Xval, Yval


def getTest_last14(shop_id, lag, is_nozero, horizon):
    # home = os.path.expanduser('~')
    # featureFolderName = (
    #     "P_%s_" + "without_zero" if is_nozero else "with_zero") % lag
    # file_name = "feature_shop_%s_%s.csv" % (
    #     shop_id, "without0" if is_nozero else "with0")
    # file_name = os.path.join(home, "Dropbox", "dataset",
    #                          "Analysis", "Features", featureFolderName, file_name)
    # featuredf = pd.read_csv(file_name)
    featuredf = getFeature(shop_id, lag, is_nozero)
    test = featuredf.iloc[featuredf.shape[0] - horizon:, :]
    Xtest = test.iloc[:, :test.shape[1] - 1]
    Ytest = test.iloc[:, test.shape[1] - 1]
    return Xtest, Ytest


def getDataSet_Tr_Va_AllShops(lag, is_nozero, val_ratio,horizon):
    shopIds = range(1, 2001)
    Xtrains = []
    Ytrains = []
    Xvals = []
    Yvals = []
    for i in shopIds:
        xt, yt, xv, yv = getDataSet_Tr_Va(i, lag, is_nozero, val_ratio,horizon)
        Xtrains.append(xt)
        Ytrains.append(yt)
        Xvals.append(xv)
        Yvals.append(yv)
    Xtrains = pd.concat(Xtrains)
    Ytrains = pd.concat(Ytrains)
    Xvals = pd.concat(Xvals)
    Yvals = pd.concat(Yvals)
    return Xtrains, Ytrains, Xvals, Yvals


def calScore(ypred, ytruth):
    '''

    :param ypred: list or np array
    :param ytruth: list or np array
    :return:
    '''
    ypred = np.array(ypred)
    ytruth = np.array(ytruth)
    dif = abs(ypred - ytruth) * 1.0
    sum_ = abs(ypred + ytruth) * 1.0
    cnt = []
    for i in range(len(dif)):
        if sum_[i]==0.0:
            tmp = 0.0
        else:
            tmp = dif[i]/sum_[i]
        cnt.append(tmp)
    return np.mean(cnt)


def baseline_val_score():
    '''
    Calculate the error of one-step ahead prediction on validation sets.
    :return:
    '''
    Xt, yt, Xv, yv = getDataSet_Tr_Va_AllShops(7,True,0.25)
    Xt = Xt.iloc[:, 2:]
    neigh = KNeighborsRegressor(n_neighbors=50, weights='distance')
    print "Fitting"
    neigh.fit(Xt, yt)
    ypred = neigh.predict(Xv.iloc[:, 2:])
    print calScore(ypred, yv)


def getRegressor(n_neighbors, Average_weights, lag, is_nozero, train_ratio, horizon):
    Xt, yt, Xv, yv = getDataSet_Tr_Va_AllShops(lag, is_nozero, 1.0-train_ratio, horizon)
    Xt = Xt.iloc[:, 2:]
    neigh = KNeighborsRegressor(n_neighbors=n_neighbors, weights=Average_weights)
    neigh.fit(Xt, yt)
    return neigh


def ItePredict(neight, X, num):
    '''

    :param neight: the regressor
    :param X: the feature of first day that needs to be predicted
    :param num: the number of consecutive days to be predicted
    :return: a list of predicted y values with length of num
    '''
    y = []
    X = np.array(X)
    X = X.reshape(1, -1)
    for i in range(num):
        #print X
        ypred = neight.predict(X)
        y.append(int(round(ypred[0])))
        X = X.reshape(-1,)
        tmp = list(X[1:])
        tmp.append(int(round(ypred[0])))
        X = np.array(tmp)
        X = X.reshape(1,-1)
    return y


def baseline_test_score(n_neighbors, Average_weights, lag, is_nozero, train_ratio, horizon):
    shops = range(1, 2001)
    print "Getting regressor..."
    neigh = getRegressor(n_neighbors, Average_weights, lag, is_nozero, train_ratio,horizon)
    print "Done"
    yTrue = []
    yPred = []
    for i in shops:
        #print "Predict %s"%i
        Xt, yt = getTest_last14(i, lag, is_nozero,horizon)
        #print "len of yt %s"%(len(yt))
        yTrue.append(yt)
        yp = ItePredict(neigh, Xt.iloc[0, 2:], horizon)
        #print "len of yp %s"%(len(yp))
        if len(yt)!=len(yp):
            print "found %s"%i
            sys.exit(0)
        yPred.extend(yp)
    yTrue = pd.concat(yTrue)
    tmp = calScore(yPred, yTrue)
    #print tmp
    return tmp


def GeneRes(n_neighbors, Average_weights, lag, is_nozero, train_ratio, horizon):
    shops = range(1, 2001)
    neigh = getRegressor(n_neighbors, Average_weights, lag, is_nozero, train_ratio, horizon)
    res = []
    for i in shops:
        Xt, yt = getTest_last14(i, lag, is_nozero,horizon)
        X = list(Xt.iloc[Xt.shape[0] - 1, :].values)
        X.append((yt.values)[len(yt) - 1])
        ypred = ItePredict(neigh, X[3:], horizon)
        res.append([i] + ypred)
    desFile = os.path.join(HOME,"Dropbox","dataset","Analysis","IterativeModel","Submission")
    file_name = "neghbours-%s_targetweight-%s_lag-%s_day-%s.csv"%(n_neighbors, Average_weights,lag,datetime.datetime.now().date())
    with open(os.path.join(desFile,file_name), 'w') as fw:
        for r in res:
            fw.write(','.join(map(str, r)) + '\n')

if __name__ == "__main__":
    #baseline_val_score()
    #baseline_test_score()
    #GeneRes()
    # metrics = ['distance']
    # neighbours = [1,3,9,10,25,50,75,100,125,150,175,200,225,250,275,300,400,500,600,700,800,900,1000,2000,3000,4000,5000]
    # lags = [12,13,14]
    # finalRes = []
    # for m in metrics:
    #     for l in lags:
    #         for n in neighbours:
    #             print m,l,n
    #             res = baseline_test_score(n, m, l, True, 1.0, 14)
    #             print res
    #             row = [m,l,n,res]
    #             finalRes.append(row)
    #
    # finalRes = pd.DataFrame(finalRes,columns=['metric','lag','neighbours','score'])
    # finalRes.to_csv("FinalScore.csv",index=False)
    #x,y = getTest_last14(561,7,True)
    #print x,len(x)
    #print y,len(y)
    GeneRes(50,"distance",10,True,1.0,14)
