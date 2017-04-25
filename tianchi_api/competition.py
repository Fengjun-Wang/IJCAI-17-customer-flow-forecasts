# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
import numpy as np
from system import getHome
from datetime import datetime
from features import get_shop_trend_zero_anomaly_filled
from metrics import loss
global_all_shop_trend = {}
global_shop_range = None
class ParameterError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class NewCompetitionPredictionModel(object):
	'''
	Speed is improved in this class
	In old version, the prediction is iterated for each shop. 
	Now the prediction is iterated over all shops.
	After experiment, doing prediction directly on a matrix(NxD,N:number of test samples;D:the feature dimention of each test sample.) is much faster than doing prediction 
	on each shop one by one in a for loop. 
	'''
    def __init__(self, estimator, estimator_output_length, required_prediction_length, feature_generation_functions, predictors, continuous_zero_filled_threshold=5, consider_anomaly=True, how_to_deal_with_remaining_zeros="filled"):
        self.estimator = estimator
        self.estimator_output_length = estimator_output_length
        self.required_prediction_length = required_prediction_length
        self.feature_generation_functions = feature_generation_functions
        if len(feature_generation_functions)==0:
            raise ParameterError("Parameter 'feature_generation_functions' must contain at least one feature generation function!")
        self.predictors = predictors
        self.continuous_zero_filled_threshold = continuous_zero_filled_threshold
        self.consider_anomaly = consider_anomaly
        self.how_to_deal_with_remaining_zeros = how_to_deal_with_remaining_zeros
        if self.how_to_deal_with_remaining_zeros not in ["filled", "drop"]:
            raise ParameterError("Parameter 'how_to_deal_with_remaining_zeros' must be one of 'filled'|'drop' ")
        self.feature_matrix = None
        self.cur_prediction_matrix=None
        self.all_shop_trend = {}
        self.shop_range = None
        self.shop_last14 = {}

    def set_shop_trend(self,if_last14=False):
        print "Initializing shop trends.."
        for shop_id in self.shop_range:
            if self.how_to_deal_with_remaining_zeros == "filled":
                shop_trend_df = get_shop_trend_zero_anomaly_filled(shop_id, 9223372036854775807, self.consider_anomaly)
            elif self.how_to_deal_with_remaining_zeros == "keep":
                shop_trend_df = get_shop_trend_zero_anomaly_filled(shop_id, self.continuous_zero_filled_threshold, self.consider_anomaly)
            else:
                shop_trend_df = get_shop_trend_zero_anomaly_filled(shop_id, self.continuous_zero_filled_threshold, self.consider_anomaly)
                shop_trend_df = shop_trend_df[shop_trend_df['cnt']!=0]
            if not if_last14:
                self.all_shop_trend[shop_id] = shop_trend_df
            else:
                #print shop_id,'hhhhh'
                self.all_shop_trend[shop_id] = shop_trend_df.iloc[:-14]
                self.shop_last14[shop_id] = list(shop_trend_df.iloc[-14:]['cnt'].values)
                #print self.shop_last14
        print "finish!"

    def gen_feature(self, cur_shop_trend, cur_shop_id):
        ini_df = self.feature_generation_functions[0][0](cur_shop_trend, cur_shop_trend.shape[0]-1, cur_shop_id)
        for func in self.feature_generation_functions[1:]:
            if func[0].__doc__ == """shop_static_info""":
                ini_df = pd.merge(ini_df, func[0](cur_shop_id), how='inner', on=func[1])
            else:
                ini_df = pd.merge(ini_df, func[0](cur_shop_trend, cur_shop_trend.shape[0]-1, cur_shop_id), how='inner', on=func[1])
        return ini_df

    def do_prediction_matrix(self):
        if self.predictors is None:
            self.cur_prediction_matrix = self.estimator.predict(self.feature_matrix)
        else:
            self.cur_prediction_matrix = self.estimator.predict(self.feature_matrix[self.predictors])
        self.cur_prediction_matrix[self.cur_prediction_matrix<0] = 1.0
        if self.cur_prediction_matrix.ndim == 1:
            self.cur_prediction_matrix = self.cur_prediction_matrix.reshape(-1, 1)

    def update_shop_trends(self):
        for i,shop in enumerate(self.shop_range):
            cur_shop_trend = self.all_shop_trend[shop]
            cur_pred = self.cur_prediction_matrix[i]
            cur_pred = list(cur_pred)
            days = pd.date_range(cur_shop_trend['day'].iloc[-1], periods=len(cur_pred)+1)
            days = [ii.date() for ii in days[1:]]
            extend_df = pd.DataFrame({'day':days,'cnt':cur_pred})
            cur_shop_trend = pd.concat([cur_shop_trend,extend_df],ignore_index=True)[['day','cnt']]
            self.all_shop_trend[shop] = cur_shop_trend

    def get_feature_matrix(self):
        res = []
        for shop in self.shop_range:
            cur_shop_trend = self.all_shop_trend[shop]
            res.append(self.gen_feature(cur_shop_trend,shop))
        self.feature_matrix = pd.concat(res,ignore_index=True)

    def do_competition(self, how="iterative",save_to_file="./for_submission.csv", shoprange=range(1,2001)):
        self.shop_range = shoprange
        if how =='iterative':
            if self.required_prediction_length % self.estimator_output_length == 0:
                rounds = self.required_prediction_length / self.estimator_output_length
            else:
                rounds = self.required_prediction_length // self.estimator_output_length + 1
            self.set_shop_trend(if_last14=False)
            preds = []
            for i in range(rounds):
                print "round %s"%i
                self.get_feature_matrix()
                self.do_prediction_matrix()
                preds.append(self.cur_prediction_matrix)
                self.update_shop_trends()
            res = np.round(np.hstack(preds)[:,:self.required_prediction_length]).astype(int)

        if how =='copy':
            rounds = self.required_prediction_length//self.estimator_output_length
            remain = self.required_prediction_length%self.estimator_output_length
            self.set_shop_trend(if_last14=False)
            self.get_feature_matrix()
            self.do_prediction_matrix()
            preds = [self.cur_prediction_matrix]*rounds
            preds.append(self.cur_prediction_matrix[:,:remain])
            res = np.round(np.hstack(preds)).astype(int)

        res = np.hstack([np.array(shoprange).reshape(-1, 1), res])
        with open(save_to_file, 'a+') as fw:
            for row in res:
                fw.write(','.join(map(str, row)) + '\n')

    def do_last14_test(self,how='iterative',save_to_file_shop_loss='./shop_loss.csv',save_to_file_day_loss='./day_loss.csv',save_to_file_shop_day_loss='./shop_day_loss.csv',shoprange=range(1,2001)):
        self.shop_range=shoprange
        if how =='iterative':
            if self.required_prediction_length % self.estimator_output_length == 0:
                rounds = self.required_prediction_length / self.estimator_output_length
            else:
                rounds = self.required_prediction_length // self.estimator_output_length + 1
            self.set_shop_trend(if_last14=True)
            preds = []
            for i in range(rounds):
                print "round %s"%i
                self.get_feature_matrix()
                self.do_prediction_matrix()
                preds.append(self.cur_prediction_matrix)
                self.update_shop_trends()
            res = np.round(np.hstack(preds)[:,:self.required_prediction_length]).astype(int)

        if how =='copy':
            rounds = self.required_prediction_length//self.estimator_output_length
            remain = self.required_prediction_length%self.estimator_output_length
            self.set_shop_trend(if_last14=True)
            self.get_feature_matrix()
            self.do_prediction_matrix()
            preds = [self.cur_prediction_matrix]*rounds
            preds.append(self.cur_prediction_matrix[:,:remain])
            res = np.round(np.hstack(preds)).astype(int)
        truth = []
        for s in self.shop_range:
            truth.append(self.shop_last14[s])
        truth = np.array(truth)
        all_loss = loss(res,truth)
        restmp = [[0,all_loss]]
        for i,s in enumerate(self.shop_range):
            restmp.append([s,loss(res[i],self.shop_last14[s])])
        with open(save_to_file_shop_loss,'a+') as fw:
            fw.write('shop_id,loss\n')
            for row in restmp:
                fw.write(','.join(map(str, row)) + '\n')
        restmp = []
        for i in xrange(self.required_prediction_length):
            pre_day = res[:,i]
            tru_day = truth[:,i]
            restmp.append([i+1,loss(pre_day,tru_day)])
        with open(save_to_file_day_loss,'a+') as fw:
            fw.write('day,loss\n')
            for row in restmp:
                fw.write(','.join(map(str, row)) + '\n')
        restmp = []
        for i in xrange(len(self.shop_range)):
            cur = [self.shop_range[i]]
            cur_pred = res[i]
            cur_truth = truth[i]
            for day in xrange(self.required_prediction_length):
                cur.append(loss([cur_pred[day]],[cur_truth[day]]))
            restmp.append(cur)
        with open(save_to_file_shop_day_loss,'a+') as fw:
            fw.write(','.join(['shop_id']+['day_'+str(i+1) for i in xrange(self.required_prediction_length)])+'\n')
            for row in restmp:
                fw.write(','.join(map(str, row)) + '\n')




class NewIterativePredictionModel(object):
    def __init__(self, estimator, estimator_output_length, required_prediction_length, feature_generation_functions, predictors, continuous_zero_filled_threshold=5, consider_anomaly=True,shop_range=range(1,2001)):
        self.estimator = estimator
        self.estimator_output_length = estimator_output_length
        self.required_prediction_length = required_prediction_length
        self.feature_generation_functions = feature_generation_functions
        if len(self.feature_generation_functions)==0:
            raise ParameterError("Parameter 'feature_generation_functions' must contain at least one feature generation function!")
        self.predictors = predictors[:]
        self.continuous_zero_filled_threshold = continuous_zero_filled_threshold
        self.consider_anomaly = consider_anomaly
        self.shop_range = shop_range
        self.feature_matrix = None
        self.cur_prediction_matrix=None
        self.shop_id_df = None
        self.all_shop_trend = {}
        self.load_all_shop_trend()

    def get_shop_id_key(self,shopid,day):
        return str(shopid)+str(day)

    def get_shopid_day(self,cur_row_feature):
        shop_id = int(cur_row_feature['shop_id'])
        tar_1_day = cur_row_feature['day']
        day_success = False
        try:
            tmp = datetime.strptime(tar_1_day, '%Y-%m-%d').date()
            tar_1_day = tmp
            day_success = True
        except:
            pass
        if not day_success:
            try:
                tar_1_day = tar_1_day.date()
            except:
                pass
        return shop_id, tar_1_day

    def load_all_shop_trend(self):
        print "loading trend database..."
        global global_all_shop_trend,global_shop_range
        if self.shop_range!=global_shop_range:
            for i in self.shop_range:
                global_all_shop_trend[i] = get_shop_trend_zero_anomaly_filled(i, self.continuous_zero_filled_threshold, self.consider_anomaly)
        global_shop_range = self.shop_range
        print "finish loading!"

    def gen_feature_row(self, cur_row):
        shopid,tar_1_day = self.get_shopid_day(cur_row)
        newkey = self.get_shop_id_key(shopid,tar_1_day)
        cur_shop_trend = self.all_shop_trend[newkey]
        ini_df = self.feature_generation_functions[0][0](cur_shop_trend, cur_shop_trend.shape[0]-1, shopid)
        for func in self.feature_generation_functions[1:]:
            if func[0].__doc__ == """shop_static_info""":
                # print "YESSTATIC"
                ini_df = pd.merge(ini_df, func[0](shopid), how='inner', on=func[1])
            else:
                ini_df = pd.merge(ini_df, func[0](cur_shop_trend, cur_shop_trend.shape[0]-1, shopid), how='inner', on=func[1])
        return ini_df

    def do_prediction_matrix(self):
        self.cur_prediction_matrix = self.estimator.predict(self.feature_matrix[self.predictors])
        self.cur_prediction_matrix[self.cur_prediction_matrix < 0] = 1.0
        if self.cur_prediction_matrix.ndim == 1:
            self.cur_prediction_matrix = self.cur_prediction_matrix.reshape(-1,1)

    def update_shop_trends(self):
        for i in xrange(self.shop_id_df.shape[0]):
            cur_row_feature = self.shop_id_df.iloc[i]
            shop_id, tar_1_day = self.get_shopid_day(cur_row_feature)
            new_key = self.get_shop_id_key(shop_id,tar_1_day)
            cur_shop_trend = self.all_shop_trend[new_key]
            cur_pred = self.cur_prediction_matrix[i]
            cur_pred = list(cur_pred)
            days = pd.date_range(cur_shop_trend['day'].iloc[-1], periods=len(cur_pred)+1)
            days = [i.date() for i in days[1:]]
            extend_df = pd.DataFrame({'day':days,'cnt':cur_pred})
            cur_shop_trend = pd.concat([cur_shop_trend,extend_df],ignore_index=True)[['day','cnt']]
            self.all_shop_trend[new_key] = cur_shop_trend

    def set_shop_trend(self, shop_id_df):
        print "Initially setting shop trend.."
        global global_all_shop_trend
        for i in xrange(shop_id_df.shape[0]):
            cur_row_feature = shop_id_df.iloc[i]
            shop_id, tar_1_day = self.get_shopid_day(cur_row_feature)
            cur_shop_trend = global_all_shop_trend[shop_id]
            delta_days = tar_1_day - cur_shop_trend['day'].iloc[0]
            tmp = cur_shop_trend.iloc[:delta_days.days]
            new_key = self.get_shop_id_key(shop_id,tar_1_day)
            self.all_shop_trend[new_key] = tmp.iloc[-40:]
        print "Finish initializing shop trend!"

    def get_feature_matrix(self):
        res = []
        for i in xrange(self.shop_id_df.shape[0]):
            cur_row = self.shop_id_df.iloc[i]
            res.append(self.gen_feature_row(cur_row))
        self.feature_matrix = pd.concat(res,ignore_index=True)

    def do_iterative_prediction(self, shop_id_df, how='iterative'):
        self.shop_id_df = shop_id_df
        if how =='iterative':
            if self.required_prediction_length % self.estimator_output_length == 0:
                rounds = self.required_prediction_length / self.estimator_output_length
            else:
                rounds = self.required_prediction_length // self.estimator_output_length + 1
            self.set_shop_trend(self.shop_id_df)
            preds = []
            for i in range(rounds):
                print "round %s"%i
                self.get_feature_matrix()
                self.do_prediction_matrix()
                preds.append(self.cur_prediction_matrix)
                self.update_shop_trends()
            return np.round(np.hstack(preds)[:,:self.required_prediction_length]).astype(int)
        if how =='copy':
            rounds = self.required_prediction_length//self.estimator_output_length
            remain = self.required_prediction_length%self.estimator_output_length
            self.set_shop_trend(self.shop_id_df)
            self.get_feature_matrix()
            self.do_prediction_matrix()
            preds = [self.cur_prediction_matrix]*rounds
            preds.append(self.cur_prediction_matrix[:,:remain])
            return np.round(np.hstack(preds)).astype(int)


class CompetitionPredictionModel(object):
    def __init__(self, estimator, estimator_output_length, required_prediction_length, feature_generation_functions, predictors, continuous_zero_filled_threshold=5, consider_anomaly=True, how_to_deal_with_remaining_zeros="Filled"):
        self.estimator = estimator
        self.estimator_output_length = estimator_output_length
        self.required_prediction_length = required_prediction_length
        self.feature_generation_functions = feature_generation_functions
        if len(feature_generation_functions)==0:
            raise ParameterError("Parameter 'feature_generation_functions' must contain at least one feature generation function!")
        self.predictors = predictors
        self.continuous_zero_filled_threshold = continuous_zero_filled_threshold
        self.consider_anomaly = consider_anomaly
        self.how_to_deal_with_remaining_zeros = how_to_deal_with_remaining_zeros
        if self.how_to_deal_with_remaining_zeros not in ["keep", "filled", "drop"]:
            raise ParameterError("Parameter 'how_to_deal_with_remaining_zeros' must be one of 'keep'|'filled'|'drop' ")
        self.shop_trend = None
        self.cur_feature = None
        self.cur_prediction = None
        self.shop_id = None
        print "Warning!!!!!"
        print "CompetitionPredictionModel is not recommended. You should try using NewCompetitionModel"

    def get_shop_trend(self):
        if self.how_to_deal_with_remaining_zeros == "filled":
            shop_trend_df = get_shop_trend_zero_anomaly_filled(self.shop_id, 9223372036854775807, self.consider_anomaly)
        elif self.how_to_deal_with_remaining_zeros == "keep":
            shop_trend_df = get_shop_trend_zero_anomaly_filled(self.shop_id, self.continuous_zero_filled_threshold, self.consider_anomaly)
        else:
            shop_trend_df = get_shop_trend_zero_anomaly_filled(self.shop_id, self.continuous_zero_filled_threshold, self.consider_anomaly)
            shop_trend_df = shop_trend_df[shop_trend_df['cnt']!=0]
        self.shop_trend = shop_trend_df

    def gen_feature(self, index):
        ini_df = self.feature_generation_functions[0][0](self.shop_trend, index, self.shop_id)
        for func in self.feature_generation_functions[1:]:
            if func[0].__doc__ == """shop_static_info""":
                # print "YESSTATIC"
                ini_df = pd.merge(ini_df, func[0](self.shop_id), how='inner', on=func[1])
            else:
                ini_df = pd.merge(ini_df, func[0](self.shop_trend, index, self.shop_id), how='inner', on=func[1])
        self.cur_feature = ini_df
        #print self.cur_feature

    def do_prediction(self):
        if self.predictors is None:
            self.cur_prediction_matrix = self.estimator.predict(self.feature_matrix)
        else:
            self.cur_prediction_matrix = self.estimator.predict(self.feature_matrix[self.predictors])
        self.cur_prediction[self.cur_prediction<0] = 1.0
        #self.cur_prediction = np.round(self.cur_prediction).astype(int)

    def update_shop_trend(self):
        shape = self.cur_prediction.shape
        if len(shape)==2:
            tmp = self.cur_prediction[0]
        else:
            tmp = self.cur_prediction
        res = list(tmp)
        days = pd.date_range(start=self.shop_trend['day'].iloc[-1], periods=len(tmp)+1)
        days = [i.date() for i in days[1:]]
        extend_df = pd.DataFrame({'day':days,'cnt':res})
        self.shop_trend = pd.concat([self.shop_trend,extend_df],ignore_index=True)[['day','cnt']]

    def set_shop_id(self, id):
        self.shop_id = id

    def predict_shop(self, shop_id, how="iterative"):
        self.shop_id = shop_id
        self.get_shop_trend()
        if how=="iterative":
            if self.required_prediction_length%self.estimator_output_length==0:
                rounds = self.required_prediction_length/self.estimator_output_length
            else:
                rounds = self.required_prediction_length//self.estimator_output_length + 1
            #print "rounds is ", rounds
            res = []
            for i in range(rounds):
                self.gen_feature(self.shop_trend.shape[0]-1)
                self.do_prediction()
                self.update_shop_trend()
                shape = self.cur_prediction.shape
                if len(shape) == 2:
                    tmp = self.cur_prediction[0]
                else:
                    tmp = self.cur_prediction
                #tmp[tmp<0]=1.0
                tmp = np.round(tmp).astype(int)
                res.extend(tmp)
            res = res[0:self.required_prediction_length]
            return res

        if how=='copy':
            res = []
            rounds = self.required_prediction_length//self.estimator_output_length
            remain = self.required_prediction_length%self.estimator_output_length
            self.gen_feature(self.shop_trend.shape[0] - 1)
            self.do_prediction()
            shape = self.cur_prediction.shape
            if len(shape) == 2:
                tmp = self.cur_prediction[0]
            else:
                tmp = self.cur_prediction
            tmp = np.round(tmp).astype(int)
            for i in xrange(rounds):
                res.extend(tmp)
            for j in xrange(remain):
                res.append(tmp[j])
            return res

    def do_competition(self, how="iterative",save_to_file="./for_submission.csv",shoprange=range(1,2001)):
        res = []
        for i in shoprange:
            print i
            cur=[i]
            cur.extend(self.predict_shop(i,how))
            res.append(cur)
        with open(save_to_file, 'a+') as fw:
            for line in res:
                fw.write(','.join(map(str,line))+'\n')

class IterativePredictionModel(object):
    def __init__(self, estimator, estimator_output_length, required_prediction_length, feature_generation_functions, predictors, continuous_zero_filled_threshold=5, consider_anomaly=True,shop_range=xrange(1,2001)):
        self.estimator = estimator
        self.estimator_output_length = estimator_output_length
        self.required_prediction_length = required_prediction_length
        self.feature_generation_functions = feature_generation_functions
        if len(self.feature_generation_functions)==0:
            raise ParameterError("Parameter 'feature_generation_functions' must contain at least one feature generation function!")
        self.predictors = predictors[:]
        self.continuous_zero_filled_threshold = continuous_zero_filled_threshold
        self.consider_anomaly = consider_anomaly
        self.shop_range = shop_range
        self.shop_trend = None
        self.cur_feature = None
        self.cur_prediction = None
        self.shop_id = None
        self.all_shop_trend = {}
        self.load_all_shop_trend()
        print "Warning!!!!!"
        print "IterativePredictionModel is not recommended. You should try using NewIterativePredictionModel"

    def load_all_shop_trend(self):
        print "loading trend database..."
        for i in self.shop_range:
            self.all_shop_trend[i] = get_shop_trend_zero_anomaly_filled(i, self.continuous_zero_filled_threshold, self.consider_anomaly)
        print "finish loading!"

    def gen_feature(self, index):
        ini_df = self.feature_generation_functions[0][0](self.shop_trend, index, self.shop_id)
        for func in self.feature_generation_functions[1:]:
            if func[0].__doc__ == """shop_static_info""":
                # print "YESSTATIC"
                ini_df = pd.merge(ini_df, func[0](self.shop_id), how='inner', on=func[1])
            else:
                ini_df = pd.merge(ini_df, func[0](self.shop_trend, index, self.shop_id), how='inner', on=func[1])
        # ini_df = self.feature_generation_functions[0](self.shop_trend, index)
        # for func in self.feature_generation_functions[1:]:
        #     if func.__doc__ == """shop_static_info""":
        #         ini_df.extend(func(self.shop_id))
        #     else:
        #         ini_df.extend(func(self.shop_trend, index))
        self.cur_feature = ini_df

    def do_prediction(self):
        self.cur_prediction = self.estimator.predict(self.cur_feature[self.predictors])
        self.cur_prediction[self.cur_prediction < 0] = 1.0

    def update_shop_trend(self):
        shape = self.cur_prediction.shape
        if len(shape)==2:
            tmp = self.cur_prediction[0]
        else:
            tmp = self.cur_prediction
        res = list(tmp)
        days = pd.date_range(start=self.shop_trend['day'].iloc[-1], periods=len(res)+1)
        days = [i.date() for i in days[1:]]
        extend_df = pd.DataFrame({'day':days,'cnt':res})
        self.shop_trend = pd.concat([self.shop_trend,extend_df],ignore_index=True)[['day','cnt']]

    def set_shop_trend(self, cur_row_feature):
        shop_id = int(cur_row_feature['shop_id'])
        self.shop_id = shop_id
        tar_1_day = cur_row_feature['day']
        day_success = False
        try:
            tmp = datetime.strptime(tar_1_day, '%Y-%m-%d').date()
            tar_1_day = tmp
            day_success = True
        except:
            pass
        if not day_success:
            try:
                tar_1_day = tar_1_day.date()
            except:
                pass
        cur_shop_trend = self.all_shop_trend[shop_id]
        delta_days = tar_1_day - cur_shop_trend['day'].iloc[0]
        self.shop_trend = cur_shop_trend.iloc[:delta_days.days]

    def predict_cur_row(self,cur_row_feature, how = 'iterative'):
        if how=='iterative':
            if self.required_prediction_length % self.estimator_output_length == 0:
                rounds = self.required_prediction_length / self.estimator_output_length
            else:
                rounds = self.required_prediction_length // self.estimator_output_length + 1
            self.set_shop_trend(cur_row_feature)
            res = []
            for i in range(rounds):
                self.gen_feature(self.shop_trend.shape[0] - 1)
                self.do_prediction()
                self.update_shop_trend()
                shape = self.cur_prediction.shape
                if len(shape) == 2:
                    tmp = self.cur_prediction[0]
                else:
                    tmp = self.cur_prediction
                tmp = np.round(tmp).astype(int)
                res.extend(tmp)
            res = res[0:self.required_prediction_length]
            return res
        if how=='copy':
            res = []
            rounds = self.required_prediction_length//self.estimator_output_length
            remain = self.required_prediction_length%self.estimator_output_length
            self.set_shop_trend(cur_row_feature)
            self.gen_feature(self.shop_trend.shape[0] - 1)
            self.do_prediction()
            shape = self.cur_prediction.shape
            if len(shape) == 2:
                tmp = self.cur_prediction[0]
            else:
                tmp = self.cur_prediction
            tmp = np.round(tmp).astype(int)
            for i in xrange(rounds):
                res.extend(tmp)
            for j in xrange(remain):
                res.append(tmp[j])
            return res


    def do_iterative_prediction(self, cur_trend_df,how='iterative'):
        res = []
        print "doing iterative prediction for "+str(cur_trend_df.shape[0])
        for i in xrange(cur_trend_df.shape[0]):
            print i+1, "remaing:"+str(cur_trend_df.shape[0]-i-1)
            res.append(self.predict_cur_row(cur_trend_df.iloc[i],how))
        print "finish iterative prediction!"
        return res



if __name__=="__main__":
    #a = Iterative_prediction()
    pass