import sys
path = 'D:/Program Files/libsvm-weights-3.24/python'
sys.path.append(path)
import numpy as np
from svmutil import *


y, x = svm_read_problem('D:/Program Files/libsvm-weights-3.24/heart_scale')
W = [1] * len(y)
W[0] = 20
W[1] = 10
W[2] = 5.5

prob = svm_problem(W, y, x)
param = svm_parameter('-s 3 -c 5 -h 0')
CV_ACC = svm_train(W, y, x, '-t 2 -c 5 -g 0.1 -v 5')