#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:05:46 2019

@author: chrisbartel
"""
import numpy as np
from sklearn.metrics import confusion_matrix, r2_score

def _make_binary_labels(data, thresh):
    return [1 if v <= thresh else 0 for v in data]

class MyStats(object):
    
    def __init__(self, actual, pred, 
                 percentiles=[1, 10, 25, 50, 75, 90, 99],
                 stability_thresholds=[0, 0.025, 0.5, 0.1, 0.2]):
        if len(actual) != len(pred):
            raise ValueError
        self.actual = actual
        self.pred = pred
        self.percentiles = percentiles
        self.stability_thresholds = stability_thresholds

    @property
    def errors(self):
        a, p = self.actual, self.pred
        return [a[i] - p[i] for i in range(len(a))]
    
    @property
    def abs_errors(self):
        errors = self.errors
        return [abs(e) for e in errors]

    @property
    def sq_errors(self):
        errors = self.errors
        return [e**2 for e in errors]
    
    @property
    def mean_error(self):
        return np.mean(self.errors)
    
    @property
    def mean_abs_error(self):
        return np.mean(self.abs_errors)
    
    @property
    def root_mean_sq_error(self):
        return np.sqrt(np.mean(self.sq_errors))
    
    @property
    def median_error(self):
        return np.median(self.errors)
    
    @property
    def median_abs_error(self):
        return np.median(self.abs_errors)
    
    @property
    def root_median_sq_error(self):
        return np.sqrt(np.median(self.sq_errors))
    
    @property
    def r2(self):
        return r2_score(self.actual, self.pred)
    
    @property
    def per_errors(self):
        percentiles = self.percentiles
        errors = self.errors
        return {p : np.percentile(errors, percentiles) for p in percentiles}

    @property
    def per_abs_errors(self):
        percentiles = self.percentiles
        errors = self.abs_errors
        return {p : np.percentile(errors, percentiles) for p in percentiles}
    
    @property
    def per_sq_errors(self):
        percentiles = self.percentiles
        errors = self.sq_errors
        return {p : np.sqrt(np.percentile(errors, percentiles)) for p in percentiles}
    
    @property
    def regression_stats(self):
        return {'abs' : {'mean' : self.mean_abs_error,
                         'median' : self.median_abs_error,
                         'per' : self.per_abs_errors},
                'raw' : {'mean' : self.mean_error,
                         'median' : self.median_error,
                         'per' : self.per_errors},
                'sq' : {'mean' : self.root_mean_sq_error,
                        'median' : self.root_median_sq_error,
                        'per' : self.per_sq_errors},
                'r2' : self.r2}
                
    def confusion(self, thresh):
        actual = _make_binary_labels(self.actual, thresh)
        pred = _make_binary_labels(self.pred, thresh)
        labels = ['tn', 'fp', 'fn', 'tp']
        return dict(zip(labels, confusion_matrix(actual, pred).ravel())) 
    
    def classification_scores(self, thresh):
        confusion = self.confusion['thresh']
        tn, fp, fn, tp = [confusion[stat] for stat in ['tn', 'afp', 'fn', 'tp']]
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        f1 = 2*(prec*rec)/(prec+rec)
        acc = (tp+tn)/(tp+tn+fp+fn)
        fpr = fp/(fp+tn)
        return {'precision' : prec,
                'recall' : rec,
                'f1' : f1,
                'accuracy' : acc,
                'fpr' : fpr}
    
    @property
    def classification_stats(self): 
        threshs = self.stability_thresholds
        return {thresh : {'raw' : self.classification(thresh),
                          'scores' : self.classification_scores(thresh)} for thresh in threshs}