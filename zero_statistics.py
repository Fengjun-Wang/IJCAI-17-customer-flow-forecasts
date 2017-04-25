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
from tianchi_api.system import getHome
import anamoly as ana

def get_original_pay_trend(shop_id):
    """
    :param shop_id:
    :return: a dataframe with columns ‘day' and 'cnt'
    """
    HOME = getHome()
    PayTrendFolder = os.path.join(HOME, "Dropbox", "dataset", "Analysis", "PayTrend")
    FileNameTemp = "CustomerFlow_%s.csv"
    csvFileName = os.path.join(PayTrendFolder, FileNameTemp % shop_id)
    shopTrend = pd.read_csv(csvFileName, header=None, names=['time', 'cnt'], parse_dates=[0])
    shopTrend['day'] = shopTrend.apply(lambda row: row['time'].date(), axis=1)
    shopTrend.drop('time',axis=1, inplace=True)
    return shopTrend

def turn_anomalies_to_zero(shop_id):
    anomalies = ana.get_anomalies()
    shop_trend = get_original_pay_trend(shop_id)
    for i in xrange(shop_trend.shape[0]):
        if shop_trend.ix[i, 'day'] in anomalies.setdefault(shop_id,set()):
            shop_trend.ix[i, 'cnt'] = 0
    return shop_trend

def get_pay_trend(shop_id, consider_anamoly):
    if not consider_anamoly:
        return get_original_pay_trend(shop_id)
    else:
        return turn_anomalies_to_zero(shop_id)

def get_zero_period_info(shop_id, consider_anamoly):
    """
    get the zero period info for each shop
    :return:
    """
    shop_trend = get_pay_trend(shop_id, consider_anamoly)
    already = False
    res = []
    for i in xrange(shop_trend.shape[0]):
        if shop_trend['cnt'].iloc[i] == 0:
            if already:
                cnt += 1
            else:
                already = True
                cnt = 1
                start_day = shop_trend['day'].iloc[i]
        else:
            if already:
                already = False
                end_day = shop_trend['day'].iloc[i-1]
                res.append([start_day, end_day, cnt])
                cnt = 0
    if already:
        res.append([start_day, shop_trend['day'].iloc[shop_trend.shape[0]-1], cnt])
    return res


def get_zerodays_to_be_filled(shop_id, threshold, consider_anamoly):
    """

    :param threshold: only consider those continuous zero-days with length <= the threshold
    :param consider_anamoly: whether consider the anamoly, if yes the anamolies will be regarded as zero days
    :return:
    """
    zero_periods = get_zero_period_info(shop_id, consider_anamoly)
    zero_periods_filtered = [i for i in zero_periods if i[2]<=threshold]
    if len(zero_periods_filtered)==0:
        return set()
    date_range = pd.date_range(zero_periods_filtered[0][0], zero_periods_filtered[0][1])
    for rec in zero_periods_filtered:
        date_range = date_range.union(pd.date_range(rec[0],rec[1]))
    return date_range





if __name__ == '__main__':
    #print get_zero_period_info(28)
    #app = QApplication(sys.argv)
    for i in range(1, 2001):
        click_data_show(i)
