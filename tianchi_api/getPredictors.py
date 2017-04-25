# -*- coding: utf-8 -*-
def predictors_WeatherAirTem(tar):
    predictors_wat = []
    days = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    remove_list = [i for i in days if (i not in tar)]
    for day in remove_list:
        predictors_wat.extend(['aqi_tar%s'%day, 'high_temp_tar%s'%day, 'low_temp_tar%s'%day, '晴天_tar%s'%day, '多云_tar%s'%day, '阴天_tar%s'%day, '小雨_tar%s'%day, '中雨_tar%s'%day, '大雨_tar%s'%day, '阵雨_tar%s'%day, '暴雨_tar%s'%day, '雾霾_tar%s'%day, '小雪_tar%s'%day, '中雪_tar%s'%day, '大雪_tar%s'%day, '阵雪_tar%s'%day, '沙尘_tar%s'%day])
    # print predictors_wat
    return predictors_wat

# predictors_WeatherAirTem([1,2])