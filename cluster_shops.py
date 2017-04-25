#!/usr/bin/python
#-*- coding: UTF-8 -*-
# import csv
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import GenFeatures as gf
import pickle


def norm():
    # normalization_5 = ['per_pay', 'comment_cnt']
    # normalization:
    max_value = shop_info_list['per_pay'].max()
    min_value = shop_info_list['per_pay'].min()
    shop_info_list['per_pay'] = 5* (shop_info_list['per_pay'] - min_value) / (max_value - min_value)
    # shop_info_list['comment_cnt'] = shop_info_list['comment_cnt'].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    max_value = shop_info_list['comment_cnt'].max()
    min_value = shop_info_list['comment_cnt'].min()
    shop_info_list['comment_cnt'] = (shop_info_list['comment_cnt'] - min_value) / (max_value - min_value)
    # print shop_info_list[:10]

def cluster_algo(algorithm_choose, shop_info_list,round):
    scores = []
    if algorithm_choose == 'DBSCAN':
        colName = ['n_clusters', 'silhouette_avg', 'eps_setting', 'min_samples_setting']
        des_scores = r'/home/fengjun/Dropbox/dataset/Analysis/Features/Cluster_shops/record_score_dbscan.csv'
        for eps_setting in range_eps:
            for min_samples_setting in range_min_samples:
                models = DBSCAN(eps=eps_setting, min_samples=min_samples_setting).fit(shop_info_list)
                cluster_labels = models.labels_
                try:
                    silhouette_avg = silhouette_score(shop_info_list, cluster_labels)
                    sample_silhouette_values = silhouette_samples(shop_info_list, cluster_labels)
                except:
                    silhouette_avg = 0
                n_clusters = len(set(list(models.labels_)))
                print set(list(models.labels_))
                scores.append([n_clusters, silhouette_avg, eps_setting, min_samples_setting])

    if algorithm_choose == "KMeans":
        colName = ['n_clusters', 'silhouette_avg']
        outFolder_pre = r'/home/fengjun/Dropbox/dataset/Analysis/Features/Cluster_shops/norm_kmeans/round' + str(round)
        import os
        try:
            #print "making.."
            os.makedirs(outFolder_pre)
        except:
            pass
        des_scores = outFolder_pre + r'/' + r'record_score_kmeans.csv'
        for n_clusters in range_n_clusters:
            models = KMeans(n_clusters=n_clusters).fit(shop_info_list)
            cluster_labels = models.labels_
            silhouette_avg = silhouette_score(shop_info_list, cluster_labels)
            scores.append([n_clusters, silhouette_avg])
            sample_silhouette_values = silhouette_samples(shop_info_list, cluster_labels)
            labels_dict = {}

            with open(outFolder_pre + r'/'  + str(n_clusters) +r'.csv', 'a+') as fw:
                for i in xrange(0, len(cluster_labels)):
                    labels_dict.setdefault(cluster_labels[i], []).append(i+1)
                    fw.write(str(cluster_labels[i]))
                    fw.write('\n')
            pickle.dump(labels_dict,
                       open(outFolder_pre + r'/'  + str(n_clusters) +r'.pickle', 'wb'))
            #
            # if n_clusters == 14:
            #     print labels_dict

            # print cluster_labels[:]
            draw_silhouette(algorithm_choose, n_clusters, sample_silhouette_values, cluster_labels, silhouette_avg)


    scores = pd.DataFrame(scores, columns=colName)
    scores.to_csv(des_scores, index=False)

    # return n_clusters, sample_silhouette_values, cluster_labels, silhouette_avg



def draw_silhouette(algorithm_choose, n_clusters, sample_silhouette_values, cluster_labels, silhouette_avg):
    #draw:
    fig, ax1 = plt.subplots()
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(shop_info_list) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    if algorithm_choose == 'KMeans':
        plt.savefig(r'/home/fengjun/Dropbox/dataset/Analysis/Features/Cluster_shops/norm_kmeans/n='+str(n_clusters)+'.png')

    if algorithm_choose == 'DBSCAN':
        plt.savefig(r'/home/fengjun/Dropbox/dataset/Analysis/Features/Cluster_shops/norm_dbscan/n='+str(n_clusters)+'.png')



filename = open(r'/home/fengjun/Dropbox/dataset/Analysis/Features/shop_info_feature.csv', 'r')
shop_info_list = pd.read_csv(filename)
shop_info_list = shop_info_list.iloc[:,1:]
norm()
algorithm_choose = 'KMeans'
# range_eps = [0.3, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
range_eps = [11]
range_min_samples = range(20, 100, 10)
range_n_clusters = range(2, 20)
# range_n_clusters = [3]
for round in xrange(0,10):
    cluster_algo(algorithm_choose, shop_info_list, round)

# predictors.extend(['score','comment_cnt','shop_level','上海','杭州','北京','广州','南京','武汉','深圳','温州','苏州','宁波','福州','成都','厦门','绍兴','无锡','济南','金华','青岛','合肥','常州','其他城市','美食','超市便利店','休闲娱乐','医疗健康','美发/美容/美甲','购物','休闲茶饮','超市','休闲食品','烘焙糕点','快餐','小吃','中餐','火锅','汤/粥/煲/砂锅/炖菜','便利店','其他美食','网吧网咖','烧烤','药店','美容美发','本地购物','个人护理','饮品/甜点','超市','奶茶','生鲜水果','面包','西式快餐','其它小吃','东北菜','中式快餐','麻辣烫/串串香','粥','蛋糕','便利店','西餐','米粉/米线','川味/重庆火锅','川菜','面点','冰激凌','网吧网咖','其它快餐,咖啡厅,粤菜,其它烘焙糕点,中式烧烤,江浙菜,零食,砂锅/煲类/炖菜,日韩料理,西北菜,其它地方菜,其它休闲食品,药店,海鲜,咖啡,其它火锅,其他餐饮美食,湖北菜,自助餐,美食特产,美容美发,香锅/烤鱼,台湾菜,闽菜,湘菜,熟食,其它烧烤,上海本帮菜,本地购物,个人护理])