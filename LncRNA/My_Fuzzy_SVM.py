import sys
path = 'D:/Program Files/libsvm_weights-3.23/python'
sys.path.append(path)
import membership
from svmutil import *
from sklearn.base import BaseEstimator, ClassifierMixin

class FSVM_Classifier(BaseEstimator, ClassifierMixin):  
    """Fuzzy SVM Classifier"""

    def __init__(self, C = 1, gamma = 0.5, membership = 'None', W = [], kernel = 'rbf'):
        """
        Called when initializing the classifier
        """
        self.C = C
        self.gamma = gamma
        self.membership = membership
        self.W = W
        self.kernel = kernel
        self.delta = 0.001

    def cal_membership(self, X, y):
        if self.membership == 'SVDD':
            W = membership.SVDD_membership(X, y, g = self.gamma, C = self.C)
        elif self.membership == 'None':
            W = []
        elif self.membership == 'precomputed':
            W = self.W
        elif self.membership == 'FSVM_2':
            W = membership.FSVM_2_membership(X, y, self.delta, membership.gaussian, g = self.gamma)
        return W

    def fit(self, X, y):
        """
        This should fit classifier.

        """
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

