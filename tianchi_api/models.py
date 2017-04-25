import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
class Ensemble_Stacking(object):

    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models
    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=2016))
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i   , clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], len(folds)))
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(1)
        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred

# class Multiout_Model(object):
#     def __init__(self, *clfs):
#         self.clfs = clfs
#         #self.predictions = None
#
#     def predict(self,X):
#         predictions = []
#         for clf in self.clfs:
#             cur_pred = clf.predict(X)
#             if cur_pred.ndim == 1:
#                 cur_pred = cur_pred.reshape(-1,1)
#             predictions.append(cur_pred)
#         #self.predictions = np.hstack(predictions)
#         return np.hstack(predictions)

class Multiout_Model(object):
    def __init__(self, clfs, predictors):
        self.clfs = clfs
        self.predictors = predictors

    def predict(self,X):
        predictions = []
        for clf,predictor in zip(self.clfs,self.predictors):
            cur_pred = clf.predict(X[predictor])
            if cur_pred.ndim == 1:
                cur_pred = cur_pred.reshape(-1,1)
            predictions.append(cur_pred)
        return np.hstack(predictions)


class XGB_Wraper(object):
    def __init__(self, xgb_instance):
        self.xgb = xgb_instance

    def predict(self, X_test):
        return self.xgb.predict(xgb.DMatrix(X_test.values, feature_names=X_test.columns.values))

#clf1 = XGB_Wraper(clf1)
#clf2 = XGB_Wraper(clf2)
#clfs_list=[clf1,clf2]
#clf_total7 = Multiout_Model(*clfs_list)
