class RF(object):
    def predict(self,X):
        return [0]*14
from competition import *
from features import *
# fgfs = [get_lag_days_for_competition(), get_mean_lag_values_for_competition(),
#             get_std_lag_values_for_competition(), get_mean_diff_lag_values_for_competition(),
#             get_difference_lag_values_for_competition(), get_difference_lag_values_for_competition(diff_order=2),
#             get_week_of_day_for_competition(),
#             get_holiday_for_target_for_competition(), get_shop_static_info_for_competition()]
fgfs = [get_lag_days_for_competition(),
        get_mean_lag_values_for_competition(),
        get_hmean_lag_values_for_competition(),
        get_std_lag_values_for_competition(),
        get_median_lag_values_for_competition(),
        get_mean_for_alllag_days_for_competition(),
        get_hmean_for_alllag_days_for_competition(),
        get_std_for_alllag_days_for_competition(),
        get_median_for_alllag_days_for_competition(),
        get_skewness_for_alllag_days_for_competition(),
        get_kurtosis_for_alllag_days_for_competition(),
        get_mean_diff_lag_values_for_competition(),
        get_difference_lag_values_for_competition(),
        get_difference_lag_values_for_competition(diff_order=2),
        get_ratio_lag_values_for_competition(),
        get_ratio_lag_values_for_competition(ratio_order=2),
        get_month_of_year_for_competition(),
        get_day_of_month_for_competition(),
        get_week_of_day_for_competition(),
        get_holiday_for_target_for_competition(),
        get_weather_aqi_for_target_for_competition(),
        get_shop_static_info_for_competition()]

# clf = RF()
# model = NewCompetitionPredictionModel(clf,14,14,fgfs,None, 5, True, 'filled')
# model.shop_range = range(1,2001)
# model.set_shop_trend(if_last14=False)
# model.get_feature_matrix()
# model.feature_matrix.to_csv('Offline_feature_new_features_newcompetition.csv',index=False)
clf = RF()
model = CompetitionPredictionModel(clf,14,14,fgfs,None, 5, True, 'filled')
res = []
for i in xrange(1,2001):
    print i
    model.set_shop_id(i)
    model.get_shop_trend()
    model.gen_feature(model.shop_trend.shape[0]-1)
    res.append(model.cur_feature)
res = pd.concat(res,ignore_index=True)
res.to_csv("Offline_feature_new_features_competition.csv",index=False)