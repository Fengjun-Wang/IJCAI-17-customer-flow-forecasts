# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.join("/home/hao", "Dropbox", "dataset", "Scripts"))
import pandas as pd
import numpy as np
from datetime import datetime
from tianchi_api.features import get_shop_trend_zero_anomaly_filled
from tianchi_api.system import getHome
import matplotlib.pyplot as plt
def get_pred_for_shops(pred_file):
    #print pred_file
    res ={}
    with open(pred_file,'r') as fp:
        lines = fp.readlines()
        for l in lines:
            terms = map(int,l.strip('\r\n').split(','))
            shop = terms[0]
            pred = terms[1:]
            res[shop] = pred
    return res
#shop_preds_xgboost = get_pred_for_shops(pred_file = os.path.join(getHome(),"Dropbox",'dataset','Analysis','competition','xgBoost_comp_iterLength_7mode_filled_tarLength_14_iterOrC_copy_0.08607781.csv'))
#shop_preds_local = get_pred_for_shops(pred_file = os.path.join(getHome(),"Dropbox",'dataset','Scripts','LocalLearning','neighbours_10000_alg_ETR_estimator_500_mss_2_msl_1_copy_7x2_drop_xiaohaozi_localLearning714_0.08547515.csv'))
#shop_records = set()


def concat_pred_to_original(shop_range, continuous_zero_filled_threshold, consider_anomaly, pred_file,savefolder):
    shop_preds = get_pred_for_shops(pred_file)
    for shop in shop_range:
        print shop
        cur_shop_trend = get_shop_trend_zero_anomaly_filled(shop, continuous_zero_filled_threshold, consider_anomaly)
        cur_shop_pred = shop_preds[shop]
        pred_days = pd.date_range('2016-11-01',periods=14)
        pred_days = [ii.date() for ii in pred_days]
        extend_df = pd.DataFrame({'day': pred_days, 'cnt': cur_shop_pred})
        plt.figure(figsize=(20,10))
        plt.plot(cur_shop_trend['day'],cur_shop_trend['cnt'],'r.-')
        plt.plot(extend_df['day'],extend_df['cnt'],'g.-')
        plt.grid()
        filename = "pred_visualization_%s.png"%shop
        try:
            os.makedirs(savefolder)
        except:
            pass
        filename = os.path.join(savefolder,filename)
        plt.savefig(filename)
        plt.close()

def writefile():
    global shop_records
    with open('shops_recorded.txt','w') as fw:
        for s in shop_records:
            fw.write("%s\n"%s)

def compare_preds(shop_id, continuous_zero_filled_threshold=5, consider_anomay=True):
    global shop_records
    def on_pick(event):
        artist = event.artist
        if shop_id not in shop_records:
            print "add %s"%shop_id
            shop_records.add(shop_id)
        else:
            print "delete %s"%shop_id
            shop_records.remove(shop_id)
        writefile()
    cur_shop_trend = get_shop_trend_zero_anomaly_filled(shop_id, continuous_zero_filled_threshold, consider_anomay)
    pred_days = pd.date_range('2016-11-01', periods=14)
    pred_days = [ii.date() for ii in pred_days]
    xgboost = shop_preds_xgboost[shop_id]
    local = shop_preds_local[shop_id]
    fig, axarr = plt.subplots(2, sharex=True,figsize=(20, 10))
    tolerance = 5
    axarr[0].plot(cur_shop_trend['day'],cur_shop_trend['cnt'],'r.-',picker=tolerance)
    axarr[0].plot(pred_days,xgboost,'g.-',picker=tolerance)
    axarr[0].set_title('%s xgboost'%shop_id)
    axarr[0].grid(True)
    axarr[1].plot(cur_shop_trend['day'],cur_shop_trend['cnt'],'r.-')
    axarr[1].plot(pred_days,local,'g.-')
    axarr[1].set_title('%s local'%shop_id)
    axarr[1].grid(True)
    fig.canvas.callbacks.connect('pick_event', on_pick)
    plt.show()

def get_substitutes():
    filename = os.path.join(getHome(),"Dropbox",'dataset','Scripts','VisualizePredictionResult','shops_recorded.txt')
    shop_set = set()
    with open(filename,'r') as fp:
        lines = fp.readlines()
        for l in lines:
            shop_set.add(int(l.strip('\r\n')))
    return shop_set

def merge_xgboost_local_substitute():
    xgboost = get_pred_for_shops(pred_file = os.path.join(getHome(),"Dropbox",'dataset','Analysis','competition','xgBoost_comp_iterLength_7mode_filled_tarLength_14_iterOrC_copy_0.08607781.csv'))
    local = get_pred_for_shops(pred_file = os.path.join(getHome(),"Dropbox",'dataset','Scripts','LocalLearning','neighbours_10000_alg_ETR_estimator_500_mss_2_msl_1_copy_7x2_drop_xiaohaozi_localLearning714_0.08547515.csv'))
    shop_set = get_substitutes()
    for shop in local:
        if shop in shop_set:
            local[shop] = xgboost[shop]
    with open('merge_xgboost_local.csv','w') as fw:
        for s in xrange(1,2001):
            fw.write(','.join(map(str,[s]+local[s]))+'\n')



if __name__=='__main__':
    pred_file = os.path.join(getHome(),"Dropbox",'dataset','Scripts','LocalLearning','Ext_LocalKernel_weight+114_77copy_0.08390_1.1_0.0832.csv')
    #print "Here"
    #pred_file = os.path.join(getHome(),"Dropbox",'dataset','Analysis','competition','features_final','xgBoost_comp_iterLength_7mode_filled_tarLength_14_iterOrC_iterative_77copy.csv')
    #print pred_file
    savefolder = os.path.join(getHome(), 'Dropbox', 'dataset', 'Analysis', 'Pred_Visulization','BestScore')
    concat_pred_to_original(range(1,2001),5,True,pred_file,savefolder)
    #for s in range(1,2001):
        #compare_preds(s)
    #merge_xgboost_local_substitute()