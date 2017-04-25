# -*- coding: utf-8 -*-
'''
This file is used to plot and save the customer flow trend as well as the view trend for each shop_id in shop_info. For those missing days we regard it as 0.
'''
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import sys
import os
import platform
import numpy as np

def getHome():
    """
    return the home directory according to the platform
    :return:
    """
    system = platform.system()
    if system.startswith("Lin"):
        HOME = os.path.expanduser('~')
    elif system.startswith("Win"):
        pass
        #HOME to be set
    else:
        print "Unknown platform"
        sys.exit(0)
    return HOME


def get_conn_postgres():
    dbStr = "dbname='{dbname}' user='{user}' host='{host}' password='{password}'"
    dbCre = {'dbname': 'CustomerFlow', 'user': 'hao',
             'host': 'localhost', 'password': '520'}
    try:
        conn = psycopg2.connect(dbStr.format(**dbCre))
    except:
        print "I am unable to connect to the database"
        sys.exit(0)
    return conn


def get_all_shopIds():
    '''
    Return all shop_ids
    should be from 1 - 2000
    '''
    conn = get_conn_postgres()
    shopIds = pd.read_sql_query('select distinct shop_id from user_pay;', conn)
    conn.close()
    return shopIds.iloc[:, 0]


def gen_customer_flow_per_shop():
    '''
    Generate the customer flow for each shop_id, for those missing days the value is regarded as 0
    '''
    des_folder = "../Analysis/PayTrend"
    try:
        os.makedirs(des_folder)
    except:
        print des_folder," existed!"
    #shopids = get_all_shopIds()
    #shopids = shopids.values
    #shopids.sort()
    shopids = range(1,2001)
    conn = get_conn_postgres()
    state = "select * from user_pay where shop_id=%s;"
    for sid in shopids:
        print "Processing: ", sid
        query_state = state % sid
        pdf_pay = pd.read_sql_query(query_state, conn)
        pdf_pay['day'] = pdf_pay.apply(
            lambda row: row['time_stamp'].date(), axis=1)
        day_cnt = pdf_pay['day'].value_counts()
        whole_period = pd.date_range(min(day_cnt.index), max(day_cnt.index))
        day_cnt = day_cnt.reindex(whole_period, fill_value=0)
        day_cnt.to_csv(os.path.join(des_folder, "CustomerFlow_%s.csv" % sid))
        day_cnt.plot(figsize=(15, 8))
        plt.xlabel("Days")
        plt.ylabel("Customer Flow")
        plt.title("CustomerFlow shop_%s" % sid)
        plt.grid()
        plt.savefig(os.path.join(des_folder, "CustomerFlow_%s.png" % sid))
        plt.close()
    conn.close()
    # plt.show()
def compareViewPay():
    '''
    Draw the view history and pay history for each shop_id on the same graph using scatter plotting.
    0 for those missing days
    '''
    des_folder = "../Analysis/PayViewCmp"
    try:
        print "making folder"
        os.makedirs(des_folder)
    except:
        print "Already exists"
    shop_ids = range(1,2001)
    conn = get_conn_postgres()
    state = "select * from user_view where shop_id=%s;"
    for sid in shop_ids:
        print "Processing: ", sid
        query_state = state % sid
        pdf_view = pd.read_sql_query(query_state, conn)
        pdf_pay = pd.read_csv('../Analysis/PayTrend/CustomerFlow_%s.csv'%sid,names=['customerFlow'],index_col=0)
        pdf_pay.index = pd.to_datetime(pdf_pay.index)
        fig, plotXY = plt.subplots(1, sharex = True,figsize=(15,8))
        plotXY.plot(pdf_pay.index.values, pdf_pay['customerFlow'].values,'g.')
        plotXY.set_ylabel("CustomerFlow")
        plotXY.grid() 
        if pdf_view.shape[0]!=0:
            pdf_view['day'] = pdf_view.apply(
            lambda row: row['time_stamp'].date(), axis=1)
            day_cnt = pdf_view['day'].value_counts()
            whole_period = pd.date_range(min(day_cnt.index), max(day_cnt.index))
            day_cnt = day_cnt.reindex(whole_period, fill_value=0)
            day_cnt.to_csv(os.path.join(des_folder, "View_%s.csv" % sid))
            day_cnt = pd.DataFrame(day_cnt)
            day_cnt.columns = ['customerFlow']

#plotXY[0].set_title(title_set)
                  
#plotXY[0].plot(data_draw_x, data[feature + '_mean_200'], color = 'r', label = 'sliding mean 200')            
            plot2 = plotXY.twinx()
#s2 = data[maintenance_code]
            plot2.plot(day_cnt.index.values, day_cnt['customerFlow'].values, 'r.')
#plot2.set_ylim(plot2.get_ylim()[::1])
            plot2.set_ylabel('View')
        #plt.show()
        plt.savefig(os.path.join(des_folder, "CustomerFlow_View_%s.png" % sid))
        plt.close()
    conn.close()

def get_orign_payTrend(shopId):
    """
    Return the pay trend data frame for shop indicated by shopId.
    The data fram is based on the csv frame in the folder:
    /home/Dropbox/dataset/Analysis/PayTrend/
    """
    HOME = getHome()
    source_folder = os.path.join(HOME, "Dropbox", "dataset", "Analysis","PayTrend")
    file_template = "CustomerFlow_%s.csv"%shopId
    file_ = os.path.join(source_folder,file_template)
    df = pd.read_csv(file_, header=None, names=['day','customer_flow'], parse_dates=[0])
    return df
    
def get_missingdays_statistics(shopId):
    """
    Get the statistics of the missing days for each shop.
    The ratio of the missing days for each shop.
    The ratio of weekdays in thos missing days.
    :return:
    """
    #HOME = getHome()
    #source_folder = os.path.join(HOME, "Dropbox", "dataset", "Analysis","PayTrend")
    #file_template = "CustomerFlow_%s.csv"%shopId
    #file_ = os.path.join(source_folder,file_template)
    #df = pd.read_csv(file_, header=None, names=['day','customer_flow'], parse_dates=[0])
    df = get_orign_payTrend(shopId)
    df['weekday_indicator'] = df.apply(lambda row:row['day'].weekday_name,axis=1)
    res_names = ['shopId','start_day','end_day','total_days','missing_days','ratio_of_missingdays','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    missdays = df[df['customer_flow']==0]
    res = [shopId]
    res.append(df.iloc[0,0].date())
    res.append(df.iloc[df.shape[0]-1,0].date())
    res.append(df.shape[0])
    res.append(missdays.shape[0])
    res.append(missdays.shape[0]*1.0/float(df.shape[0]))
    weekday_cnt = missdays['weekday_indicator'].value_counts()
    for d in res_names[-7:]:
        try:
            res.append(weekday_cnt[d]/float(missdays.shape[0]))
        except:
            res.append(0.0)
    res = [res]
    return pd.DataFrame(res,columns=res_names)

def get_missingdays_statistics_for_all_shops():
    res = []
    for i in range(1,2001):
        print i
        res.append(get_missingdays_statistics(i))
    res = pd.concat(res)
    HOME = getHome()
    source_folder = os.path.join(HOME, "Dropbox", "dataset", "Analysis")
    desfile = os.path.join(source_folder, "missing_days_statistics.csv")
    res.to_csv(desfile, index=False)
# def get_filled_value(method, shop_trend_df):
#     def filled_by_mean
#     if method=='mean':
#         return
def getValue(method,i,cusFlow):
    #if cusFlow[i]!=0:
        #return cusFlow[i]
    if method=='mean':
        for_mean = []
        if i<=3:
            recent_range=range(0,i)+range(i+1,9)
        elif i>=len(cusFlow)-4:
            recent_range = range(len(cusFlow)-9,i)+range(i+1,len(cusFlow))
        else:
            recent_range = range(i-4,i)+range(i+1,i+5)
        ii = 3
        jj = 4
        while ((ii>=0 and jj<=7) and len(for_mean)<4):
            if cusFlow[recent_range[ii]]!=0:
                if len(for_mean)<4:
                    for_mean.append(cusFlow[recent_range[ii]])
            ii-=1
            if cusFlow[recent_range[jj]]!=0:
                if len(for_mean)<4:
                    for_mean.append(cusFlow[recent_range[jj]])
            jj+=1
        ii = 1
        for_mean_sameday=[]
        break2 = False
        break1 = False
        while (len(for_mean_sameday)<4):
            if i+ii*7<len(cusFlow):
                if cusFlow[i+ii*7]!=0:
                    for_mean_sameday.append(cusFlow[i+ii*7])
            else:
                break1=True
            if i-ii*7>=0:
                if cusFlow[i-ii*7]!=0:
                    if len(for_mean_sameday)<4:
                        for_mean_sameday.append(cusFlow[i-ii*7])
            else:
                break2=True
            if break1 and break2:
                break
            ii+=1
        resfinal = for_mean+for_mean_sameday
        return np.mean(resfinal)

    if method=='knn':
        pass
def get_Customer_flow_per_shop_missingdays_handled(method,shopId, threshold_for_zero_length, consider_anamoly):
    """
    This function does the same as gen_customer_flow_per_shop() except that for those missing days we do not simply regard them as 0.
    We adopt some strategy to interpolate those values according to the parameter method
    """
    from zero_statistics import get_zerodays_to_be_filled
    from zero_statistics import get_pay_trend
    from zero_statistics import get_original_pay_trend
    days_to_be_filled = get_zerodays_to_be_filled(shopId, threshold_for_zero_length, consider_anamoly)
    shop_trend_df = get_pay_trend(shopId, consider_anamoly)
    addCol = []
    cusFlow = list(shop_trend_df['cnt'].values)
    for i in range(len(cusFlow)):
        if shop_trend_df['day'].iloc[i] in days_to_be_filled:
            addCol.append(getValue(method, i, cusFlow))
        else:
            addCol.append(cusFlow[i])
    original_trend = get_original_pay_trend(shopId)
    original_trend['filled_customer_flow'] = addCol
    #from itertools import cycle, islice
    #my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(original_trend)))

    #original_trend.plot(color=my_colors,figsize=(15, 8))
    #plt.grid()
    HOME = getHome()
    source_folder = os.path.join(HOME, "Dropbox", "dataset", "Analysis","PayTrend_Filled_threshold_%s_%s_anomaly"%(threshold_for_zero_length,"consider" if consider_anamoly else "not_consider"))
    try:
        os.makedirs(source_folder)
    except:
        pass
    #plt.savefig(os.path.join(source_folder,"Customer_Flow_zeroFilled_shop_%s.png"%shopId))
    original_trend[['day','cnt','filled_customer_flow']].to_csv(os.path.join(source_folder,"Customer_Flow_zeroFilled_shop_%s.csv"%shopId),index=False,header=False)
    #plt.close()





if __name__ == '__main__':
    #gen_customer_flow_per_shop()
    #compareViewPay()
    #get_missingdays_statistics_for_all_shops()
    for i in xrange(1,2001):
        print i
        get_Customer_flow_per_shop_missingdays_handled('mean',i, 1, True)
    