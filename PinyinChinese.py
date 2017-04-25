# -*- coding: utf-8 -*-
'''
This file is used to change the Chinese characters in the shop_info into Pinyin.
'''
import pandas as pd
#import matplotlib.pyplot as plt
from pypinyin import lazy_pinyin


def get_pinyin(chs_name):
    '''
    return the pinyin in str type for  given  Chinese words in type str
    '''
    return reduce(lambda x, y: x + y,
                  map(lambda x: x.capitalize(), lazy_pinyin(chs_name.decode('utf-8')))).encode('utf-8')


def Chinese2Pinyin():
    '''
    Get dictionaries containing conversions from Chinese words to Pinyin 
    '''
    shop_info = '../shop_info.txt'
    shop_info = pd.read_csv(shop_info, header=None, names=[
                            'shop_id', 'city_name', 'location_id', 'per_pay', 'score', 'comment_cnt', 'shop_level', 'cate_1_name', 'cate_2_name', 'cate_3_name'])
    city_cnt = shop_info['city_name'].value_counts()
    cityName = {}
    for city in city_cnt.index:
        cityPinyin = get_pinyin(city)
        cityName[city] = cityPinyin
    cate1Name = {}
    cate1Cnt = shop_info['cate_1_name'].value_counts()
    for cate in cate1Cnt.index:
        catePinyin = get_pinyin(cate)
        cate1Name[cate] = catePinyin

    cate2Name = {}
    cate2Cnt = shop_info['cate_2_name'].value_counts()
    for cate in cate2Cnt.index:
        catePinyin = get_pinyin(cate)
        cate2Name[cate] = catePinyin

    cate3Name = {}
    cate3Cnt = shop_info['cate_3_name'].value_counts()
    for cate in cate3Cnt.index:
        catePinyin = get_pinyin(cate)
        cate3Name[cate] = catePinyin
    return cityName, cate1Name, cate2Name, cate3Name


def modifyShop_info():
    cityName, cate1Name, cate2Name, cate3Name = Chinese2Pinyin()

    shop_info = '../shop_info.txt'
    with open('../shop_info_pinyin.txt', 'w') as fw:
        shop_info = open(shop_info, 'r')
        for line in shop_info.readlines():
            line = line.strip('\r\n')
            terms = line.split(',')
            if terms[1] != '':
                terms[1] = cityName[terms[1]]

            if terms[7] != '':
                terms[7] = cate1Name[terms[7]]

            if terms[8] != '':
                terms[8] = cate2Name[terms[8]]

            if terms[9] != '':
                terms[9] = cate3Name[terms[9]]
            fw.write(','.join(terms) + '\n')
        shop_info.close()


if __name__ == '__main__':
    modifyShop_info()
