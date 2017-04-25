# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
import numpy as np
from system import getHome
from datetime import datetime
from datetime import timedelta
from scipy import stats
import pickle
def get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold=5, consider_anomaly=True):
    HOME = getHome()
    source_folder = os.path.join(HOME, "Dropbox", "dataset", "Analysis", "PayTrend_Filled_threshold_%s_%s_anomaly" % (
        continuous_zero_filled_threshold, "consider" if consider_anomaly else "not_consider"))
    source_file = os.path.join(source_folder,"Customer_Flow_zeroFilled_shop_%s.csv"%shop_id)
    if not os.path.exists(source_file):
        #print 100101
        print "Generating: %s\n"%source_file
        sys.path.append(os.path.join(HOME, "Dropbox", "dataset", "Scripts"))
        import TrendForEveryShop as tfes
        tfes.get_Customer_flow_per_shop_missingdays_handled('mean', shop_id, continuous_zero_filled_threshold, consider_anomaly)
    #print source_file
    df = pd.read_csv(source_file, header=None, names=['time', 'orig_cnt', 'cnt'], parse_dates=[0])
    df.drop('orig_cnt', inplace=True, axis=1)
    df['day'] = df.apply(lambda row: row['time'].date(), axis=1)
    df.drop('time', axis=1, inplace=True)
    return df[['day','cnt']]


def get_shop_feature(shop_id, feature_generation_functions=[], ifSave=True, save_to_file=None, existed_detection=True):
    """
    :param shop_id:
    :param feature_generation_functions: 特征生成函数列表，
    :param ifSave:
    :param save_to_file: 请确保文件所在目录已经存在，只有当ifsave设置为True的时候才有用
    :param existed_detection: 如果是True,则会检测save_to_file是否存在，如果存在就直接返回，如果是False,则每次都重新生成特征，并根据ifsave和save_to_file进行保存与否
    :return: return the feature data frame
    """
    if existed_detection:
        if os.path.exists(save_to_file):
            print "File %s detected, loading from it."%save_to_file
            return pd.read_csv(save_to_file)
    if len(feature_generation_functions) == 0:
        print "No feature generation functions supplied!"
        sys.exit(0)
    ini_df = feature_generation_functions[0][0](shop_id)
    for func in feature_generation_functions[1:]:
        #print func
        ini_df = pd.merge(ini_df, func[0](shop_id), how='inner', on=func[1])
    if ifSave:
        if save_to_file is None:
            print "Please indicate the file path which you want to write feature matrix into!"
        else:
            ini_df.to_csv(save_to_file, index=False)
    return ini_df

def get_target_variables(continuous_zero_filled_threshold=5, consider_anomaly=True, drop_invalids=True, invalid_values=[0], output_length=14):
    """
    得到目标值，默认长度为14,就是你之前的Tar
    :param continuous_zero_filled_threshold: 这个天数以下的连续０会被填充，填入-1表示所有的０都不填充
    :param consider_anomaly: 是否考虑去除异常点,就是那些突然特别高的点,如果是，这些点会先被弄成０
    :param drop_invalids: 是否要去除无效值，比如无效值为０,则14个目标中有一个为０，这个样本就不考虑。
    :param invalid_values: 哪些是无效值，默认为０，接受一个list,可以是多个无效值，比如[0,1,2,3]
    :param output_length: 目标长度
    :return:
    """
    def inner_function(shop_id):
        df = get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold, consider_anomaly)
        res = []
        for i in xrange(df.shape[0]-output_length+1):
            cur_rec = [shop_id, df['day'].iloc[i]]
            cur_rec.extend(df['cnt'].iloc[i:i+output_length])
            res.append(cur_rec)
        column_names = ['shop_id', 'day'] + ["Tar_"+str(i) for i in xrange(1, output_length+1)]
        res = pd.DataFrame(res, columns=column_names)
        if drop_invalids:
            invalid_values_set = set(invalid_values)
            res = res.loc[~(res.isin(invalid_values_set).any(axis=1)), :]
        return res
    return (inner_function, ['shop_id','day'])

def get_lag_days(continuous_zero_filled_threshold=5, consider_anomaly=True, drop_invalids=True, invalid_values=[0],lag=21):
    """
    得到过去若干天的客流量
    :param continuous_zero_filled_threshold: 这个天数以下的连续０会被填充，填入-1表示所有的０都不填充
    :param consider_anomaly: 是否考虑去除异常点,就是那些突然特别高的点,如果是，这些点会先被弄成０
    :param drop_invalids: 是否要去除无效值，比如无效值为０,则14个目标中有一个为０，这个样本就不考虑。
    :param invalid_values: 哪些是无效值，默认为０，接受一个list,可以是多个无效值，比如[0,1,2,3]
    :param lag: 过去的天数的长度
    :return:
    """
    def inner_function(shop_id):
        df = get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold, consider_anomaly)
        res = []
        for i in xrange(lag, df.shape[0]):
            cur_rec = [shop_id, df['day'].iloc[i]]
            cur_rec.extend(df['cnt'].iloc[i-lag:i])
            res.append(cur_rec)
        column_names = ['shop_id', 'day'] + ['P_'+str(i) for i in range(lag, 0, -1)]
        res = pd.DataFrame(res, columns=column_names)
        if drop_invalids:
            invalid_values_set = set(invalid_values)
            res = res.loc[~(res.isin(invalid_values_set).any(axis=1)), :]
        return res
    return (inner_function, ['shop_id', 'day'])

def get_lag_days_for_competition(lag=21):
    def inner_function(df, index, shop_id):
        cur = [shop_id]
        curday = df['day'].iloc[index]
        tar_day1 = curday+timedelta(days=1)
        cur.append(tar_day1)
        cur.extend( list(df['cnt'].iloc[index-lag+1:index+1].values))
        column_names = ['shop_id', 'day'] + ['P_' + str(i) for i in range(lag, 0, -1)]
        res = [cur]
        res = pd.DataFrame(res, columns=column_names)
        return res
    return (inner_function, ['shop_id','day'])

def get_mean_lag_values(continuous_zero_filled_threshold=5, consider_anomaly=True, drop_invalids=True, invalid_values=[0], mean_window_size=7, lag=21):
    """
    :param continuous_zero_filled_threshold:
    :param consider_anomaly:
    :param drop_invalids:
    :param invalid_values:
    :param mean_window_size: 求平均值的滑动窗口大小
    :param lag: 过去的天数
    :return:
    """
    def inner_function(shop_id):
        df = get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold, consider_anomaly)
        i = lag + mean_window_size-1
        invalid_values_set = set(invalid_values)
        res = []
        while (i < df.shape[0]):
            past = df['cnt'].iloc[i-(lag+mean_window_size-1):i]
            if drop_invalids:
                tmp_com_result = past.isin(invalid_values_set)
                if tmp_com_result.any():
                    j = tmp_com_result.shape[0]-1
                    while (tmp_com_result.iloc[j]!=True):
                        j-=1
                    i = i-(lag+mean_window_size-1)+j+lag+mean_window_size
                    continue
            cur_rec = [shop_id, df['day'].iloc[i]]
            mean_tmp = []
            for jj in xrange(i-lag, i):
                mean_tmp.append(np.mean(df['cnt'].iloc[jj-mean_window_size+1:jj+1]))
            cur_rec.extend(mean_tmp)
            res.append(cur_rec)
            i+=1
        column_names = ['shop_id','day']
        column_names.extend(["P_%s_mean_%s"%(i,mean_window_size) for i in xrange(lag, 0, -1)])
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_mean_lag_values_for_competition(mean_window_size=7, lag=21):
    def inner_function(df, index, shop_id):
        cur = [shop_id]
        curday = df['day'].iloc[index]
        tar_day1 = curday + timedelta(days=1)
        cur.append(tar_day1)
        for i in xrange(index-lag+1,index+1):
            cur.append(np.mean(df['cnt'].iloc[i-mean_window_size+1:i+1]))
        res = [cur]
        column_names = ['shop_id','day']
        column_names.extend(["P_%s_mean_%s"%(i,mean_window_size) for i in xrange(lag, 0, -1)])
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_hmean_lag_values(continuous_zero_filled_threshold=5, consider_anomaly=True, drop_invalids=True, invalid_values=[0], hmean_window_size=7, lag=21):
    def inner_function(shop_id):
        df = get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold, consider_anomaly)
        i = lag + hmean_window_size-1
        invalid_values_set = set(invalid_values)
        res = []
        while (i < df.shape[0]):
            past = df['cnt'].iloc[i-(lag+hmean_window_size-1):i]
            if drop_invalids:
                tmp_com_result = past.isin(invalid_values_set)
                if tmp_com_result.any():
                    j = tmp_com_result.shape[0]-1
                    while (tmp_com_result.iloc[j]!=True):
                        j-=1
                    i = i-(lag+hmean_window_size-1)+j+lag+hmean_window_size
                    continue
            cur_rec = [shop_id, df['day'].iloc[i]]
            mean_tmp = []
            for jj in xrange(i-lag, i):
                mean_tmp.append(stats.hmean(df['cnt'].iloc[jj-hmean_window_size+1:jj+1]))
            cur_rec.extend(mean_tmp)
            res.append(cur_rec)
            i+=1
        column_names = ['shop_id','day']
        column_names.extend(["P_%s_hmean_%s"%(i,hmean_window_size) for i in xrange(lag, 0, -1)])
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_hmean_lag_values_for_competition(hmean_window_size=7, lag=21):
    def inner_function(df, index, shop_id):
        cur = [shop_id]
        curday = df['day'].iloc[index]
        tar_day1 = curday + timedelta(days=1)
        cur.append(tar_day1)
        for i in xrange(index-lag+1,index+1):
            cur.append(stats.hmean(df['cnt'].iloc[i-hmean_window_size+1:i+1]))
        res = [cur]
        column_names = ['shop_id','day']
        column_names.extend(["P_%s_hmean_%s"%(i,hmean_window_size) for i in xrange(lag, 0, -1)])
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_std_lag_values(continuous_zero_filled_threshold=5, consider_anomaly=True, drop_invalids=True, invalid_values=[0], std_window_size=7, lag=21):
    """

    :param continuous_zero_filled_threshold:
    :param consider_anomaly:
    :param drop_invalids:
    :param invalid_values:
    :param std_window_size: 求标准差的窗口大小
    :param lag: 过去的天数
    :return:
    """
    def inner_function(shop_id):
        df = get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold, consider_anomaly)
        i = lag + std_window_size-1
        invalid_values_set = set(invalid_values)
        res = []
        while (i < df.shape[0]):
            past = df['cnt'].iloc[i-(lag+std_window_size-1):i]
            if drop_invalids:
                tmp_com_result = past.isin(invalid_values_set)
                if tmp_com_result.any():
                    j = tmp_com_result.shape[0]-1
                    while (tmp_com_result.iloc[j]!=True):
                        j-=1
                    i = i-(lag+std_window_size-1)+j+lag+std_window_size
                    continue
            cur_rec = [shop_id, df['day'].iloc[i]]
            std_tmp = []
            for jj in xrange(i-lag, i):
                std_tmp.append(np.std(df['cnt'].iloc[jj-std_window_size+1:jj+1]))
            cur_rec.extend(std_tmp)
            res.append(cur_rec)
            i+=1
        column_names = ['shop_id','day']
        column_names.extend(["P_%s_std_%s"%(i,std_window_size) for i in xrange(lag, 0, -1)])
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_std_lag_values_for_competition(std_window_size=7, lag=21):
    def inner_function(df, index, shop_id):
        cur = [shop_id]
        curday = df['day'].iloc[index]
        tar_day1 = curday + timedelta(days=1)
        cur.append(tar_day1)
        for i in xrange(index-lag+1,index+1):
            cur.append(np.std(df['cnt'].iloc[i-std_window_size+1:i+1]))
        res = [cur]
        column_names = ['shop_id','day']
        column_names.extend(["P_%s_std_%s"%(i,std_window_size) for i in xrange(lag, 0, -1)])
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_median_lag_values(continuous_zero_filled_threshold=5, consider_anomaly=True, drop_invalids=True, invalid_values=[0], median_window_size=7, lag=21):
    def inner_function(shop_id):
        df = get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold, consider_anomaly)
        i = lag + median_window_size-1
        invalid_values_set = set(invalid_values)
        res = []
        while (i < df.shape[0]):
            past = df['cnt'].iloc[i-(lag+median_window_size-1):i]
            if drop_invalids:
                tmp_com_result = past.isin(invalid_values_set)
                if tmp_com_result.any():
                    j = tmp_com_result.shape[0]-1
                    while (tmp_com_result.iloc[j]!=True):
                        j-=1
                    i = i-(lag+median_window_size-1)+j+lag+median_window_size
                    continue
            cur_rec = [shop_id, df['day'].iloc[i]]
            mean_tmp = []
            for jj in xrange(i-lag, i):
                mean_tmp.append(np.median(df['cnt'].iloc[jj-median_window_size+1:jj+1]))
            cur_rec.extend(mean_tmp)
            res.append(cur_rec)
            i+=1
        column_names = ['shop_id','day']
        column_names.extend(["P_%s_median_%s"%(i,median_window_size) for i in xrange(lag, 0, -1)])
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_median_lag_values_for_competition(median_window_size=7, lag=21):
    def inner_function(df, index, shop_id):
        cur = [shop_id]
        curday = df['day'].iloc[index]
        tar_day1 = curday + timedelta(days=1)
        cur.append(tar_day1)
        for i in xrange(index-lag+1,index+1):
            cur.append(np.median(df['cnt'].iloc[i-median_window_size+1:i+1]))
        res = [cur]
        column_names = ['shop_id','day']
        column_names.extend(["P_%s_median_%s"%(i,median_window_size) for i in xrange(lag, 0, -1)])
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_mean_for_alllag_days(continuous_zero_filled_threshold=5, consider_anomaly=True, drop_invalids=True, invalid_values=[0], lag=21):
    def inner_function(shop_id):
        df = get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold, consider_anomaly)
        i = lag
        invalid_values_set = set(invalid_values)
        res = []
        while (i < df.shape[0]):
            past = df['cnt'].iloc[i-lag:i]
            if drop_invalids:
                tmp_com_result = past.isin(invalid_values_set)
                if tmp_com_result.any():
                    j = tmp_com_result.shape[0]-1
                    while (tmp_com_result.iloc[j]!=True):
                        j-=1
                    i = i-lag+j+lag+1
                    continue
            cur_rec = [shop_id, df['day'].iloc[i]]
            cur_rec.append(np.mean(past.values))
            res.append(cur_rec)
            i+=1
        column_names = ['shop_id','day']
        column_names.append("mean_for_past_%s_days"%lag)
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_mean_for_alllag_days_for_competition(lag=21):
    def inner_function(df,index,shop_id):
        #print df.shape,index,shop_id
        cur = [shop_id]
        curday = df['day'].iloc[index]
        tar_day1 = curday + timedelta(days=1)
        cur.append(tar_day1)
        cur.append(np.mean(df['cnt'].iloc[index-lag+1:index+1]))
        res = [cur]
        column_names = ['shop_id','day','mean_for_past_%s_days'%lag]
        return pd.DataFrame(res,columns=column_names)
    return (inner_function, ['shop_id','day'])

def get_hmean_for_alllag_days(continuous_zero_filled_threshold=5, consider_anomaly=True, drop_invalids=True, invalid_values=[0], lag=21):
    def inner_function(shop_id):
        df = get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold, consider_anomaly)
        i = lag
        invalid_values_set = set(invalid_values)
        res = []
        while (i < df.shape[0]):
            past = df['cnt'].iloc[i-lag:i]
            if drop_invalids:
                tmp_com_result = past.isin(invalid_values_set)
                if tmp_com_result.any():
                    j = tmp_com_result.shape[0]-1
                    while (tmp_com_result.iloc[j]!=True):
                        j-=1
                    i = i-lag+j+lag+1
                    continue
            cur_rec = [shop_id, df['day'].iloc[i]]
            cur_rec.append(stats.hmean(past.values))
            res.append(cur_rec)
            i+=1
        column_names = ['shop_id','day']
        column_names.append("hmean_for_past_%s_days"%lag)
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_hmean_for_alllag_days_for_competition(lag=21):
    def inner_function(df,index,shop_id):
        cur = [shop_id]
        curday = df['day'].iloc[index]
        tar_day1 = curday + timedelta(days=1)
        cur.append(tar_day1)
        cur.append(stats.hmean(df['cnt'].iloc[index-lag+1:index+1]))
        res = [cur]
        column_names = ['shop_id','day','hmean_for_past_%s_days'%lag]
        return pd.DataFrame(res,columns=column_names)
    return (inner_function, ['shop_id','day'])

def get_std_for_alllag_days(continuous_zero_filled_threshold=5, consider_anomaly=True, drop_invalids=True, invalid_values=[0], lag=21):
    def inner_function(shop_id):
        df = get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold, consider_anomaly)
        i = lag
        invalid_values_set = set(invalid_values)
        res = []
        while (i < df.shape[0]):
            past = df['cnt'].iloc[i-lag:i]
            if drop_invalids:
                tmp_com_result = past.isin(invalid_values_set)
                if tmp_com_result.any():
                    j = tmp_com_result.shape[0]-1
                    while (tmp_com_result.iloc[j]!=True):
                        j-=1
                    i = i-lag+j+lag+1
                    continue
            cur_rec = [shop_id, df['day'].iloc[i]]
            cur_rec.append(np.std(past.values))
            res.append(cur_rec)
            i+=1
        column_names = ['shop_id','day']
        column_names.append("std_for_past_%s_days"%lag)
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_std_for_alllag_days_for_competition(lag=21):
    def inner_function(df,index,shop_id):
        cur = [shop_id]
        curday = df['day'].iloc[index]
        tar_day1 = curday + timedelta(days=1)
        cur.append(tar_day1)
        cur.append(np.std(df['cnt'].iloc[index-lag+1:index+1]))
        res = [cur]
        column_names = ['shop_id','day','std_for_past_%s_days'%lag]
        return pd.DataFrame(res,columns=column_names)
    return (inner_function, ['shop_id','day'])


def get_median_for_alllag_days(continuous_zero_filled_threshold=5, consider_anomaly=True, drop_invalids=True, invalid_values=[0], lag=21):
    def inner_function(shop_id):
        df = get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold, consider_anomaly)
        i = lag
        invalid_values_set = set(invalid_values)
        res = []
        while (i < df.shape[0]):
            past = df['cnt'].iloc[i-lag:i]
            if drop_invalids:
                tmp_com_result = past.isin(invalid_values_set)
                if tmp_com_result.any():
                    j = tmp_com_result.shape[0]-1
                    while (tmp_com_result.iloc[j]!=True):
                        j-=1
                    i = i-lag+j+lag+1
                    continue
            cur_rec = [shop_id, df['day'].iloc[i]]
            cur_rec.append(np.median(past.values))
            res.append(cur_rec)
            i+=1
        column_names = ['shop_id','day']
        column_names.append("median_for_past_%s_days"%lag)
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_median_for_alllag_days_for_competition(lag=21):
    def inner_function(df,index,shop_id):
        cur = [shop_id]
        curday = df['day'].iloc[index]
        tar_day1 = curday + timedelta(days=1)
        cur.append(tar_day1)
        cur.append(np.median(df['cnt'].iloc[index-lag+1:index+1]))
        res = [cur]
        column_names = ['shop_id','day','median_for_past_%s_days'%lag]
        return pd.DataFrame(res,columns=column_names)
    return (inner_function, ['shop_id','day'])

def get_skewness_for_alllag_days(continuous_zero_filled_threshold=5, consider_anomaly=True, drop_invalids=True, invalid_values=[0], lag=21):
    def inner_function(shop_id):
        df = get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold, consider_anomaly)
        i = lag
        invalid_values_set = set(invalid_values)
        res = []
        while (i < df.shape[0]):
            past = df['cnt'].iloc[i-lag:i]
            if drop_invalids:
                tmp_com_result = past.isin(invalid_values_set)
                if tmp_com_result.any():
                    j = tmp_com_result.shape[0]-1
                    while (tmp_com_result.iloc[j]!=True):
                        j-=1
                    i = i-lag+j+lag+1
                    continue
            cur_rec = [shop_id, df['day'].iloc[i]]
            cur_rec.append(stats.skew(past.values))
            res.append(cur_rec)
            i+=1
        column_names = ['shop_id','day']
        column_names.append("skewness_for_past_%s_days"%lag)
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_skewness_for_alllag_days_for_competition(lag=21):
    def inner_function(df,index,shop_id):
        cur = [shop_id]
        curday = df['day'].iloc[index]
        tar_day1 = curday + timedelta(days=1)
        cur.append(tar_day1)
        cur.append(stats.skew(df['cnt'].iloc[index-lag+1:index+1]))
        res = [cur]
        column_names = ['shop_id','day','skewness_for_past_%s_days'%lag]
        return pd.DataFrame(res,columns=column_names)
    return (inner_function, ['shop_id','day'])

def get_kurtosis_for_alllag_days(continuous_zero_filled_threshold=5, consider_anomaly=True, drop_invalids=True, invalid_values=[0], lag=21):
    def inner_function(shop_id):
        df = get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold, consider_anomaly)
        i = lag
        invalid_values_set = set(invalid_values)
        res = []
        while (i < df.shape[0]):
            past = df['cnt'].iloc[i-lag:i]
            if drop_invalids:
                tmp_com_result = past.isin(invalid_values_set)
                if tmp_com_result.any():
                    j = tmp_com_result.shape[0]-1
                    while (tmp_com_result.iloc[j]!=True):
                        j-=1
                    i = i-lag+j+lag+1
                    continue
            cur_rec = [shop_id, df['day'].iloc[i]]
            cur_rec.append(stats.kurtosis(past.values))
            res.append(cur_rec)
            i+=1
        column_names = ['shop_id','day']
        column_names.append("kurtosis_for_past_%s_days"%lag)
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_kurtosis_for_alllag_days_for_competition(lag=21):
    def inner_function(df,index,shop_id):
        cur = [shop_id]
        curday = df['day'].iloc[index]
        tar_day1 = curday + timedelta(days=1)
        cur.append(tar_day1)
        cur.append(stats.kurtosis(df['cnt'].iloc[index-lag+1:index+1]))
        res = [cur]
        column_names = ['shop_id','day','kurtosis_for_past_%s_days'%lag]
        return pd.DataFrame(res,columns=column_names)
    return (inner_function, ['shop_id','day'])

def get_mean_diff_lag_values(continuous_zero_filled_threshold=5, consider_anomaly=True, drop_invalids=True, invalid_values=[0], mean_window_size=7, diff_gap_size=7, lag=21):
    """
    得到你的mean diff
    :param continuous_zero_filled_threshold:
    :param consider_anomaly:
    :param drop_invalids:
    :param invalid_values:
    :param mean_window_size: 平均值窗口大小
    :param diff_gap_size: 求差的间隔
    :param lag:
    :return:
    """
    def inner_function(shop_id):
        df = get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold, consider_anomaly)
        i = lag + mean_window_size+diff_gap_size-1
        invalid_values_set = set(invalid_values)
        res = []
        while (i < df.shape[0]):
            past = df['cnt'].iloc[i-(lag+diff_gap_size+mean_window_size-1):i]
            if drop_invalids:
                tmp_com_result = past.isin(invalid_values_set)
                if tmp_com_result.any():
                    j = tmp_com_result.shape[0]-1
                    while (tmp_com_result.iloc[j]!=True):
                        j-=1
                    i = i-(lag+diff_gap_size+mean_window_size-1)+j+lag+mean_window_size+diff_gap_size
                    continue
            cur_rec = [shop_id, df['day'].iloc[i]]
            mean_diff_tmp = []
            for jj in xrange(i-lag, i):
                mean_diff_tmp.append(np.mean(df['cnt'].iloc[jj-mean_window_size+1:jj+1])-np.mean(df['cnt'].iloc[jj-diff_gap_size-mean_window_size+1:jj-diff_gap_size+1]))
            cur_rec.extend(mean_diff_tmp)
            res.append(cur_rec)
            i+=1
        column_names = ['shop_id','day']
        column_names.extend(["P_%s_mean_%s_diff_%s"%(i,mean_window_size,diff_gap_size) for i in xrange(lag, 0, -1)])
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_mean_diff_lag_values_for_competition(mean_window_size=7, diff_gap_size=7, lag=21):
    def inner_function(df, index, shop_id):
        cur = [shop_id]
        curday = df['day'].iloc[index]
        tar_day1 = curday + timedelta(days=1)
        cur.append(tar_day1)
        for i in xrange(index-lag+1,index+1):
            cur.append(np.mean(df['cnt'].iloc[i-mean_window_size+1:i+1])-np.mean(df['cnt'].iloc[i-diff_gap_size-mean_window_size+1:i-diff_gap_size+1]))
        res = [cur]
        column_names = ['shop_id','day']
        column_names.extend(["P_%s_mean_%s_diff_%s"%(i,mean_window_size,diff_gap_size) for i in xrange(lag, 0, -1)])
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_difference_lag_values(continuous_zero_filled_threshold=5, consider_anomaly=True, drop_invalids=True, invalid_values=[0], diff_order=1, lag=21):
    """
    :param continuous_zero_filled_threshold:
    :param consider_anomaly:
    :param drop_invalids:
    :param invalid_values:
    :param diff_order: 差分阶数
    :param lag:
    :return:
    """
    def inner_function(shop_id):
        df = get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold, consider_anomaly)
        i = lag + diff_order
        invalid_values_set = set(invalid_values)
        res = []
        while i<df.shape[0]:
            past = df['cnt'].iloc[i-(lag+diff_order):i]
            if drop_invalids:
                tmp_com_result = past.isin(invalid_values_set)
                if tmp_com_result.any():
                    j = tmp_com_result.shape[0] - 1
                    while (tmp_com_result.iloc[j] != True):
                        j-=1
                    i = i-(lag + diff_order) + j + lag + diff_order + 1
                    continue
            cur_rec = [shop_id, df['day'].iloc[i]]
            diff_order_tmp = list(past.values)

            for j in range(diff_order):
                tmp = [diff_order_tmp[ii]-diff_order_tmp[ii-1] for ii in range(1, len(diff_order_tmp))]
                diff_order_tmp = tmp
            cur_rec.extend(diff_order_tmp)
            res.append(cur_rec)
            i += 1
            #print i
        column_names = ['shop_id', 'day']
        column_names.extend(["P_%s_difforder_%s" % (i, diff_order) for i in xrange(lag, 0, -1)])
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_difference_lag_values_for_competition(diff_order=1, lag=21):
    def inner_function(df, index, shop_id):
        cur = [shop_id]
        curday = df['day'].iloc[index]
        tar_day1 = curday + timedelta(days=1)
        cur.append(tar_day1)
        past = df['cnt'].iloc[index-(lag-1 + diff_order):index+1]
        diff_order_tmp = list(past.values)
        for j in range(diff_order):
            tmp = [diff_order_tmp[ii] - diff_order_tmp[ii - 1] for ii in range(1, len(diff_order_tmp))]
            diff_order_tmp = tmp
        cur.extend(diff_order_tmp)
        res = [cur]
        column_names = ['shop_id', 'day']
        column_names.extend(["P_%s_difforder_%s" % (i, diff_order) for i in xrange(lag, 0, -1)])
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_ratio_lag_values(continuous_zero_filled_threshold=5, consider_anomaly=True, drop_invalids=True, invalid_values=[0], ratio_order=1, lag=21):
    def inner_function(shop_id):
        df = get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold, consider_anomaly)
        i = lag + ratio_order
        invalid_values_set = set(invalid_values)
        res = []
        while i<df.shape[0]:
            past = df['cnt'].iloc[i-(lag+ratio_order):i]
            if drop_invalids:
                tmp_com_result = past.isin(invalid_values_set)
                if tmp_com_result.any():
                    j = tmp_com_result.shape[0] - 1
                    while (tmp_com_result.iloc[j] != True):
                        j-=1
                    i = i-(lag + ratio_order) + j + lag + ratio_order + 1
                    continue
            cur_rec = [shop_id, df['day'].iloc[i]]
            diff_order_tmp = list(past.values)

            for j in range(ratio_order):
                tmp = [float(diff_order_tmp[ii])/float(diff_order_tmp[ii-1]) for ii in range(1, len(diff_order_tmp))]
                diff_order_tmp = tmp
            cur_rec.extend(diff_order_tmp)
            res.append(cur_rec)
            i += 1
        column_names = ['shop_id', 'day']
        column_names.extend(["P_%s_ratioorder_%s" % (i, ratio_order) for i in xrange(lag, 0, -1)])
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_ratio_lag_values_for_competition(ratio_order=1,lag=21):
    def inner_function(df, index, shop_id):
        cur = [shop_id]
        curday = df['day'].iloc[index]
        tar_day1 = curday + timedelta(days=1)
        cur.append(tar_day1)
        past = df['cnt'].iloc[index-(lag-1 + ratio_order):index+1]
        diff_order_tmp = list(past.values)
        for j in range(ratio_order):
            tmp = [float(diff_order_tmp[ii])/float(diff_order_tmp[ii - 1]) for ii in range(1, len(diff_order_tmp))]
            diff_order_tmp = tmp
        cur.extend(diff_order_tmp)
        res = [cur]
        column_names = ['shop_id', 'day']
        column_names.extend(["P_%s_ratioorder_%s" % (i, ratio_order) for i in xrange(lag, 0, -1)])
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_month_of_year(continuous_zero_filled_threshold=5, consider_anomaly=True, drop_invalids=True, invalid_values=[0]):
    def inner_function(shop_id):
        df = get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold, consider_anomaly)
        res = []
        invalid_values_set = set(invalid_values)
        for i in xrange(df.shape[0]):
            if drop_invalids:
                if df['cnt'].iloc[i] in invalid_values_set:
                    continue
            cur_rec = [shop_id, df['day'].iloc[i]]
            tmp = [0] * 12
            tmp[df['day'].iloc[i].month-1] = 1
            cur_rec.extend(tmp)
            res.append(cur_rec)
        column_names = ['shop_id', 'day']
        column_names.extend(['month%s_of_year' % i for i in xrange(12)])
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_month_of_year_for_competition():
    def inner_function(df,index,shop_id):
        cur = [shop_id]
        curday = df['day'].iloc[index]
        tar_day1 = curday + timedelta(days=1)
        cur.append(tar_day1)
        tmp = [0]*12
        tmp[tar_day1.month-1] = 1
        cur.extend(tmp)
        res=[cur]
        column_names = ['shop_id', 'day']
        column_names.extend(['month%s_of_year' % i for i in xrange(12)])
        return pd.DataFrame(res,columns=column_names)
    return (inner_function, ['shop_id','day'])


def get_day_of_month(continuous_zero_filled_threshold=5, consider_anomaly=True, drop_invalids=True, invalid_values=[0]):
    def inner_function(shop_id):
        df = get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold, consider_anomaly)
        res = []
        invalid_values_set = set(invalid_values)
        for i in xrange(df.shape[0]):
            if drop_invalids:
                if df['cnt'].iloc[i] in invalid_values_set:
                    continue
            cur_rec = [shop_id, df['day'].iloc[i]]
            tmp = [0] * 31
            tmp[df['day'].iloc[i].day-1] = 1
            cur_rec.extend(tmp)
            res.append(cur_rec)
        column_names = ['shop_id', 'day']
        column_names.extend(['day%s_of_month' % i for i in xrange(31)])
        return pd.DataFrame(res, columns=column_names)
    return (inner_function, ['shop_id', 'day'])

def get_day_of_month_for_competition():
    def inner_function(df,index,shop_id):
        cur = [shop_id]
        curday = df['day'].iloc[index]
        tar_day1 = curday + timedelta(days=1)
        cur.append(tar_day1)
        tmp = [0]*31
        tmp[tar_day1.day-1] = 1
        cur.extend(tmp)
        res=[cur]
        column_names = ['shop_id', 'day']
        column_names.extend(['day%s_of_month' % i for i in xrange(31)])
        return pd.DataFrame(res,columns=column_names)
    return (inner_function, ['shop_id','day'])

def get_week_of_day(continuous_zero_filled_threshold=5, consider_anomaly=True, drop_invalids=True, invalid_values=[0]):
    """
    星期几
    :param continuous_zero_filled_threshold:
    :param consider_anomaly:
    :param drop_invalids:
    :param invalid_values:
    :return:
    """
    def inner_function(shop_id):
        df = get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold, consider_anomaly)
        res = []
        invalid_values_set = set(invalid_values)
        for i in xrange(df.shape[0]):
            if drop_invalids:
                if df['cnt'].iloc[i] in invalid_values_set:
                    continue
            cur_rec = [shop_id, df['day'].iloc[i]]
            tmp = [0]*7
            tmp[df['day'].iloc[i].weekday()] = 1
            cur_rec.extend(tmp)
            res.append(cur_rec)
        column_names = ['shop_id','day']
        column_names.extend(['dayOfWeek_%s'%i for i in range(7)])
        return pd.DataFrame(res,columns=column_names)
    return (inner_function,['shop_id','day'])

def get_week_of_day_for_competition():
    def inner_function(df, index, shop_id):
        cur = [shop_id]
        curday = df['day'].iloc[index]
        tar_day1 = curday + timedelta(days=1)
        cur.append(tar_day1)
        #day = pd.date_range(start=df['day'].iloc[index],periods=2)[-1].date()
        res = [0]*7
        res[tar_day1.weekday()] = 1
        cur.extend(res)
        res = [cur]
        column_names = ['shop_id','day']
        column_names.extend(['dayOfWeek_%s'%i for i in range(7)])
        return pd.DataFrame(res,columns=column_names)
    return (inner_function,['shop_id','day'])

def get_holiday_for_target(continuous_zero_filled_threshold=5, consider_anomaly=True, drop_invalids=True, invalid_values=[0], output_length=14):
    """
    目标是否是节假日
    :param continuous_zero_filled_threshold:
    :param consider_anomaly:
    :param drop_invalids:
    :param invalid_values:
    :param output_length:
    :return:
    """
    def inner_function(shop_id):
        holiday_dict = {
            2: ['2015-09-27', '2015-10-01', '2016-01-01', '2016-02-07', '2016-02-08', '2016-02-09', '2016-04-04',
                '2016-05-01',
                '2016-06-09', '2016-09-15', '2016-10-01', '2016-10-02', '2016-10-03']}
        holidaylist = [datetime.strptime(d, '%Y-%m-%d').date() for d in holiday_dict[2]]
        invalid_values_set = set(invalid_values)
        df = get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold, consider_anomaly)
        res = []
        for i in xrange(df.shape[0]-output_length+1):
            cur_rec = [shop_id, df['day'].iloc[i]]
            if drop_invalids:
                following_cnts = df['cnt'].iloc[i:i+output_length]
                following_cnts_com_result = following_cnts.isin(invalid_values_set)
                if following_cnts_com_result.any():
                    continue
            following_days = df['day'].iloc[i:i+output_length]
            holidays = [0]*output_length
            for iii in range(len(following_days)):
                #print following_days[iii], holidaylist
                if following_days.iloc[iii] in holidaylist:
                    holidays[iii]=1
            cur_rec.extend(holidays)
            res.append(cur_rec)
        column_names = ['shop_id', 'day'] + ["Holiday_"+str(i) for i in xrange(1, output_length+1)]
        res = pd.DataFrame(res, columns=column_names)
        return res
    return (inner_function, ['shop_id','day'])

def get_holiday_for_target_for_competition(output_length=14):
    def inner_function(df,index,shop_id):
        cur = [shop_id]
        curday = df['day'].iloc[index]
        tar_day1 = curday + timedelta(days=1)
        cur.append(tar_day1)
        holiday_dict = {
            2: ['2015-09-27', '2015-10-01', '2016-01-01', '2016-02-07', '2016-02-08', '2016-02-09', '2016-04-04',
                '2016-05-01',
                '2016-06-09', '2016-09-15', '2016-10-01', '2016-10-02', '2016-10-03']}
        holidaylist = [datetime.strptime(d, '%Y-%m-%d').date() for d in holiday_dict[2]]
        days = pd.date_range(start=df['day'].iloc[index],periods=output_length+1)
        days = [i.date() for i in days[1:]]
        res = [0]*output_length
        for iii in range(len(days)):
            if days[iii] in holidaylist:
                res[iii] = 1
        cur.extend(res)
        res = [cur]
        column_names = ['shop_id', 'day'] + ["Holiday_"+str(i) for i in xrange(1, output_length+1)]
        res = pd.DataFrame(res, columns=column_names)
        return res
    return (inner_function, ['shop_id','day'])

def get_shop_2_city():
    PKL_file = os.path.join(getHome(),'Dropbox','dataset','Scripts','weather','shop_2_city.pkl')
    return pickle.load(open(PKL_file,'rb'))
def get_city_to_daily_weather():
    PKL_file = os.path.join(getHome(),'Dropbox','dataset','Scripts','weather','city_to_daily_weather.pkl')
    return pickle.load(open(PKL_file,'rb'))
def get_city_to_daily_aqi():
    PKL_file = os.path.join(getHome(),'Dropbox','dataset','Scripts','weather','city_to_daily_aqi.pkl')
    return pickle.load(open(PKL_file,'rb'))
def get_weather_aqi_for_target(continuous_zero_filled_threshold=5, consider_anomaly=True, drop_invalids=True, invalid_values=[0], output_length=14):
    cityname = get_shop_2_city()
    city_weather = get_city_to_daily_weather()
    city_aqi = get_city_to_daily_aqi()
    def inner_function(shop_id):
        invalid_values_set = set(invalid_values)
        df = get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold, consider_anomaly)
        res = []
        for i in xrange(df.shape[0] - output_length + 1):
            cur_rec = [shop_id, df['day'].iloc[i]]
            if drop_invalids:
                following_cnts = df['cnt'].iloc[i:i + output_length]
                following_cnts_com_result = following_cnts.isin(invalid_values_set)
                if following_cnts_com_result.any():
                    continue
            following_days = df['day'].iloc[i:i + output_length]
            for day in following_days:
                city_name_pinyin = cityname[shop_id]
                tmpday = day
                if day not in city_aqi[city_name_pinyin].keys():
                    tmpday = tmpday + timedelta(days=1)
                    while tmpday not in city_aqi[city_name_pinyin].keys():
                        tmpday = tmpday + timedelta(days=1)
                aqi = city_aqi[city_name_pinyin][tmpday]
                tmpday = day
                if day not in city_weather[city_name_pinyin].keys():
                    tmpday = tmpday + timedelta(days=1)
                    while tmpday not in city_weather[city_name_pinyin].keys():
                        tmpday = tmpday + timedelta(days=1)
                weather = city_weather[city_name_pinyin][tmpday]
                cur_rec.extend([aqi]+weather)
            res.append(cur_rec)
        column_names = ['shop_id', 'day']
        w_aqi_names = ['aqi','high_temp','low_temp','晴天', '多云', '阴天', '小雨', '中雨', '大雨', '阵雨', '暴雨', '雾霾', '小雪', '中雪', '大雪', '阵雪', '沙尘']
        for i in xrange(1,output_length+1):
            column_names.extend([nn+"_tar"+str(i) for nn in w_aqi_names])
        res = pd.DataFrame(res, columns=column_names)
        return res
    return (inner_function, ['shop_id', 'day'])

def get_weather_aqi_for_target_for_competition(output_length=14):
    cityname = get_shop_2_city()
    city_weather = get_city_to_daily_weather()
    city_aqi = get_city_to_daily_aqi()
    def inner_function(df,index,shop_id):
        cur = [shop_id]
        curday = df['day'].iloc[index]
        tar_day1 = curday + timedelta(days=1)
        cur.append(tar_day1)
        following_days = pd.date_range(tar_day1,periods=output_length)
        following_days = [iii.date() for iii in following_days]
        for day in following_days:
            city_name_pinyin = cityname[shop_id]
            tmpday = day
            if day not in city_aqi[city_name_pinyin].keys():
                tmpday = tmpday + timedelta(days=1)
                while tmpday not in city_aqi[city_name_pinyin].keys():
                    tmpday = tmpday + timedelta(days=1)
            aqi = city_aqi[city_name_pinyin][tmpday]
            tmpday = day
            if day not in city_weather[city_name_pinyin].keys():
                tmpday = tmpday + timedelta(days=1)
                while tmpday not in city_weather[city_name_pinyin].keys():
                    tmpday = tmpday + timedelta(days=1)
            weather = city_weather[city_name_pinyin][tmpday]
            cur.extend([aqi]+weather)
        res=[cur]
        column_names = ['shop_id', 'day']
        w_aqi_names = ['aqi','high_temp','low_temp','晴天', '多云', '阴天', '小雨', '中雨', '大雨', '阵雨', '暴雨', '雾霾', '小雪', '中雪', '大雪', '阵雪', '沙尘']
        for i in xrange(1,output_length+1):
            column_names.extend([nn+"_tar"+str(i) for nn in w_aqi_names])
        res = pd.DataFrame(res, columns=column_names)
        return res
    return (inner_function, ['shop_id', 'day'])

def get_shop_static_info():
    """
    商店静态信息
    :return:
    """
    def inner_function(shop_id):
        FILE = os.path.join(getHome(), "Dropbox", "dataset", "Analysis", "Features", "shop_info_feature.csv")
        return pd.read_csv(FILE)
    return (inner_function,['shop_id'])

def get_shop_static_info_for_competition():
    def inner_function(shop_id):
        """shop_static_info"""
        FILE = os.path.join(getHome(), "Dropbox", "dataset", "Analysis", "Features", "shop_info_feature.csv")
        df = pd.read_csv(FILE)
        df = df[df['shop_id']==shop_id]
        return df
    return inner_function,['shop_id']

if __name__=="__main__":
    #get_shop_trend_zero_anomaly_filled(1,5,True)
    #print get_lag_days()[0](1)
    #print get_target_variables()[0](1)
    #print get_std_lag_values()[0](1)
    #print get_difference_lag_values()[0](1)
    #print get_week_of_day()[0](1)
    #print get_holiday_for_target()[0](1)
    #print get_shop_static_info()[0](1)
    """example usage"""
    for i in xrange(1,2001):
        print i
        Folder = './NewFeatures'
        try:
            os.makedirs(Folder)
        except:
            pass

        filename = "feature_shop_%s.csv"%i
        filename = os.path.join(Folder,filename)
        get_shop_feature(i,[get_lag_days(),
                        get_mean_lag_values(),
                        get_hmean_lag_values(),
                        get_std_lag_values(),
                        get_median_lag_values(),
                        get_mean_for_alllag_days(),
                        get_hmean_for_alllag_days(),
                        get_std_for_alllag_days(),
                        get_median_for_alllag_days(),
                        get_skewness_for_alllag_days(),
                        get_kurtosis_for_alllag_days(),
                        get_mean_diff_lag_values(),
                        get_difference_lag_values(),
                        get_difference_lag_values(diff_order=2),
                        get_ratio_lag_values(),
                        get_ratio_lag_values(ratio_order=2),
                        get_month_of_year(),
                        get_day_of_month(),
                        get_week_of_day(),
                        get_holiday_for_target(),
                        get_weather_aqi_for_target(),
                        get_shop_static_info(),
                        get_target_variables()], True, filename, True)
    """For each feature set you can set parameters as you like"""