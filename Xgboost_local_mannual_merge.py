# -*- coding: utf-8 -*-
'''
This file helps to find the shops that may get higher score in Xgboost than local learning.
Two submission files are:
/home/hao/Dropbox/dataset/Scripts/LocalLearning/neighbours_10000_alg_ETR_estimator_500_mss_2_msl_1_copy_7x2_drop_xiaohaozi_localLearning714_0.08547515.csv
/home/hao/Dropbox/dataset/Analysis/competition/xgBoost_comp_iterLength_7mode_filled_tarLength_14_iterOrC_copy_0.08607781.csv
'''

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from tianchi_api.system import getHome
import zero_statistics as zt
global_anamolies = set()
DESFOLDER = os.path.join(getHome(),"Dropbox","dataset","Analysis","AnamolyDetect")
FILENAME = "firstpart.csv"

def write_anamolies():
    try:
        os.makedirs(DESFOLDER)
    except:
        pass
    with open(os.path.join(DESFOLDER,FILENAME),'w') as fw:
        for rec in global_anamolies:
            fw.write("%s,%s,%s\n"%(rec[0],rec[1],rec[2]))

def click_data_show(shop_id):
    global global_anamolies
    def on_pick(event):
        artist = event.artist
        #xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
        x, y = artist.get_xdata(), artist.get_ydata()
        ind = event.ind

        data_point = (shop_id,x[ind[0]],y[ind[0]])
        if data_point in global_anamolies:
            text = 'Data point:(%s,%s,%s) deleted' % data_point
            print text
            global_anamolies.remove(data_point)
        else:
            text = 'Data point:(%s,%s,%s) added' % data_point
            print text
            global_anamolies.add(data_point)
        write_anamolies()
    shop_trend = zt.get_original_pay_trend(shop_id)
    fig, ax = plt.subplots(figsize=(20, 10))
    tolerance = 5
    ax.plot(shop_trend['day'], shop_trend['cnt'] ,'go--', picker=tolerance, markerfacecolor='red')
    fig.canvas.callbacks.connect('pick_event', on_pick)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    #print get_zero_period_info(28)
    #app = QApplication(sys.argv)
    for i in range(1, 2001):
        click_data_show(i)