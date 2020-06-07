import sys
path = 'D:/Program Files/libsvm_weights-3.23/python'
sys.path.append(path)
import membership
import numpy as np
from svmutil import *
from sklearn.base import BaseEstimator, ClassifierMixin

class FSVM_Classifier(BaseEstimator, ClassifierMixin):  
    """Fuzzy SVM Classifier"""

    def __init__(self, C = 1, gamma = 0.5, nu = 0.5, membership = 'None', W = [], kernel = 'rbf', proj = 'linear'):
        """
        Called when initializing the classifier
        """
        self.C = C
        self.gamma = gamma
        self.membership = membership
        self.W = W
        self.kernel = kernel
        self.delta = 0.001
        self.nu = nu
        self.proj = proj

    def cal_membership(self, X, y):
        if self.membership == 'SVDD':
            W = membership.SVDD_membership(X, y, g = self.gamma, C = self.nu, proj = self.proj)
        elif self.membership == 'None':
            W = []
        return W

    def fit(self, X, y, Weight = []):
        """
        This should fit classifier.

        """
        if self.membership == 'precomputed':
            W = Weight
        else:
            W = self.cal_membership(X, y)
        if self.kernel == 'rbf':
            prob = svm_problem(W = W, y = y, x = X)
            param = svm_parameter('-t 2 -c '+str(self.C)+' -g '+str(self.gamma) + ' -b 1 -q')
        elif self.kernel == 'precomputed':
            prob  = svm_problem(W = W, y = y, x = X, isKernel = True)
            param = svm_parameter('-t 4 -c '+str(self.C)+' -b 1 -q')
        self.model = svm_train(prob, param)
        return self

    def predict(self, X, y = []):
        self.p_label, self.p_acc, self.p_val = svm_predict(y, X, self.model, '-b 1 -q')
        return self.p_label

    def predict_proba(self, X, y = []):
        self.p_label, self.p_acc, self.p_val = svm_predict(y, X, self.model, '-b 1 -q')
        self.y_proba = np.reshape(self.p_val, (len(self.p_val), 2))
        return self.y_proba

    def decision_function(self, X, y = []):
        self.p_label, self.p_acc, self.p_val = svm_predict(y, X, self.model, '-b 1 -q')
        self.y_proba = np.reshape(self.p_val, (len(self.p_val), 2))
        return self.y_proba[:, 1]
