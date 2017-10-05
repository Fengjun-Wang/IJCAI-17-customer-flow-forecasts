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
#from zero_statistics import get_original_pay_trend
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

def get_anomalies():
    """

    :return: a dictionary containing anamolies for each shop {shop_id:Set(day)}
    """
    res = {}
    tar_file = os.path.join(DESFOLDER, FILENAME)
    with open(tar_file, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            terms = line.split(',')
            res.setdefault(int(terms[0]), set()).add(pd.to_datetime(terms[1]).date())
    return res


if __name__ == '__main__':
    #print get_zero_period_info(28)
    #app = QApplication(sys.argv)
    for i in range(1, 2001):
        click_data_show(i)