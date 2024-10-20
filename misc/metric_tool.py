import numpy as np


###################       metrics      ###################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

    def clear(self):
        self.initialized = False


###################      cm metrics      ###################
class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""
    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        # self.update(val, weight) could be used to solve data with imblanced class
        # default output is  confuse_matrix itself without modify
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val)
        return current_score

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict
    
# harmonic mean = n*(1/x1 + 1/x2 + 1/x3)
def harmonic_mean(xs):
    harmonic_mean = len(xs) / sum((x+1e-6)**-1 for x in xs)
    return harmonic_mean

def cm2F1(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)
    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)
    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)
    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    return mean_F1

'''
acc = OA = (TP + TN)/(TP + TN + FP + FN)
recall = TP/(TP + FN) recall_0(TP,FN 0) recall_1(TP,FN 1)
precision = TP/(TP + FP)
'''
def cm2score(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    # tp : (n_class,)
    tp = np.diag(hist)
    # row:the number of true class(class 1 row,add along col,the true number of class 1)
    # col:the number of class which is predicted as true(class 1 col,add along row,the predict number of class 1)
    # sum_a1,sum_a0 : (n_class,)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    # add smallest positive float number to avoid divide zero
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score(more close to 1,more better) harmonic mean of recall and precision
    F1 = 2*recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    
    # ---------------------------------------------------------------------- #
    # 2. Frequency weighted Accuracy & Mean IoU
    # https://www.jianshu.com/p/01596f4e0979
    # ---------------------------------------------------------------------- #
    # IOU ：Union is the sum of category i across rows and columns minus the duplicate TP (True Positives) part, 
    # while Intersection is the TP part, IOU is tp/(actually true + predict true - tp)
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)
    # true number of sample/total number of sample
    # replace the coefficient of MIOU with freq (which is the weight of IOU)
    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    # dict(zip()) transfer tuple to dict
    # iu precision etc is list, contain number more than one
    cls_iou = dict(zip(['iou_'+str(i) for i in range(n_class)], iu))
    cls_precision = dict(zip(['precision_'+str(i) for i in range(n_class)], precision))
    cls_recall = dict(zip(['recall_'+str(i) for i in range(n_class)], recall))
    cls_F1 = dict(zip(['F1_'+str(i) for i in range(n_class)], F1))
    # add new dict if dict duplicate, new one will replace the old
    score_dict = {'acc': acc, 'miou': mean_iu, 'mf1':mean_F1}
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)
    return score_dict

# Closure : def __fast_hist(label_gt, label_pred)
# from sklearn.metrics import confusion_matrix
# confusion_matrix(ground_truth,predict_label)
def get_confuse_matrix(num_classes, label_gts, label_preds):
    def __fast_hist(label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        # mask is a bool type array, label_gt could include multiclasses, not just two classes
        # label_gt[mask] only contain the label in reasonable field(0 ~ num_class - 1), the other part will be abandon
        mask = (label_gt >= 0) & (label_gt < num_classes)
        # mapping all types of combination between groud truth and predict label into different values,can't simply add
        # for example, the condition groud truth 0,predict label 3 and groud truth 1,predict label 2,both of them get 3
        # but they are two condition,we need one condition one value mapping
        # The corresponding combination unit has already been counted before reshaping, and the reshape op does not affect it 
        # So the output shape is n*n, but only one position(one combination) is counted
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes**2).reshape(num_classes, num_classes)
        return hist
    confusion_matrix = np.zeros((num_classes, num_classes))
    # label_gts, label_preds is numpy array, lt,lp is a single element belong to numpy.int32 array object which could be flattern()
    # however array = [1] the 1 means a integer which could not be flattern just like numpy.int32(array object)
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix


def get_mIoU(num_classes, label_gts, label_preds):
    confusion_matrix = get_confuse_matrix(num_classes, label_gts, label_preds)
    score_dict = cm2score(confusion_matrix)
    return score_dict['miou']
