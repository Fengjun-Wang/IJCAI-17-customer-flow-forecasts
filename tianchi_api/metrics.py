# -*- coding: utf-8 -*-
import numpy as np

class PredError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class DimensionError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

def get_sum(ypred, ytruth, ifchecked):
    '''
    :param ypred: np array
    :param ytruth: np array
    :return:
    '''
    if ifchecked:
        check = checkPred(ypred)
        if check==False:
            raise PredError("Predicted values must be >= 0!")
    ypred[ypred<0] = 1
    dif = abs(ypred - ytruth) * 1.0
    sum_ = abs(ypred + ytruth) * 1.0
    cnt = []
    for i in range(len(dif)):
        if sum_[i]==0.0:
            tmp = 0.0
        else:
            tmp = dif[i]/sum_[i]
        cnt.append(tmp)
    return np.sum(cnt)

def checkPred(ypred):
    check = (ypred>=0.0)
    return np.all(check)

def loss(ypred, ytruth, ifchecked=True):
    """
    :param ypred: array-like of shape = (n_samples) or (n_samples, n_outputs) Estimated target values.
    :param ytruth: array-like of shape = (n_samples) or (n_samples, n_outputs) Ground truth (correct) target values.
    :return:loss
    """
    #a= type(ytruth)
    #print type(a),a
    ypred = np.array(ypred)
    ytruth = np.array(ytruth)
    ypred_shape = ypred.shape
    ytruth_shape = ytruth.shape
    if ypred_shape != ytruth_shape:
        raise DimensionError("Dimension mismatch! prediction:%s, truth: %s"%(ypred_shape,ytruth_shape))
    if (len(ypred_shape)==1) and (len(ytruth_shape)==1):
        return get_sum(ypred, ytruth, ifchecked)/float(len(ytruth))
    elif (len(ypred_shape)==2) and (len(ytruth_shape)==2):
        total_sum = 0.0
        for i in range(len(ytruth)):
            total_sum += get_sum(ypred[i],ytruth[i], ifchecked)
        return total_sum/float(ytruth_shape[0]*ytruth_shape[1])
    else:
        raise DimensionError("Dimension of prediction and truth is limited to 1 or 2 ! The current dimension is %s!"%len(ytruth_shape))

def loss_reverse(ypred, ytruth, ifchecked=True):
    """
    :param ypred: array-like of shape = (n_samples) or (n_samples, n_outputs) Estimated target values.
    :param ytruth: array-like of shape = (n_samples) or (n_samples, n_outputs) Ground truth (correct) target values.
    :return:loss
    """
    ypred = np.array(ypred)
    ytruth = np.array(ytruth)
    ypred_shape = ypred.shape
    ytruth_shape = ytruth.shape
    if ypred_shape != ytruth_shape:
        raise DimensionError("Dimension mismatch! prediction:%s, truth: %s"%(ypred_shape,ytruth_shape))
    if (len(ypred_shape)==1) and (len(ytruth_shape)==1):
        return float(len(ytruth))/get_sum(ypred, ytruth, ifchecked)
    elif (len(ypred_shape)==2) and (len(ytruth_shape)==2):
        total_sum = 0.0
        for i in range(len(ytruth)):
            total_sum += get_sum(ypred[i],ytruth[i], ifchecked)
        return float(ytruth_shape[0]*ytruth_shape[1])/total_sum
    else:
        raise DimensionError("Dimension of prediction and truth is limited to 1 or 2 ! The current dimension is %s!"%len(ytruth_shape))
